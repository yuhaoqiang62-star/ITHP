from torch import nn
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
from transformers.models.bert.modeling_bert import BertPooler
from ITHP import ITHP
import global_configs
from global_configs import DEVICE
import torch


class ITHP_DebertaModel(DebertaV2PreTrainedModel):
    """
    支持9种消融实验变体的DeBERTa模型:
    1. T      - 仅文本 (Baseline: DeBERTa → Classifier)
    2. A      - 仅声学 (Acoustic → MLP → Classifier)
    3. V      - 仅视觉 (Visual → MLP → Classifier)
    4. T+A    - 双模态Layer 1 (Text+Acoustic → B_G → Classifier)
    5. T+V    - 双模态Layer 1 (Text+Visual → B_F → Classifier)
    6. A+V    - 双模态Layer 1 (Acoustic+Visual → B_G → Classifier)
    7. T+A+V  - 三模态flat (Concat(T,A,V) → Fusion MLP → Classifier)
    8. T+V+A  - 层次变体 (Layer1: T+V→B_G, Layer2: B_G+A→B_1)
    9. GICA   - 完整层次模型 (Layer1: T+A→B_G, Layer2: B_G+V→B_1)
    """
    
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = (
            global_configs.TEXT_DIM, global_configs.ACOUSTIC_DIM, global_configs.VISUAL_DIM
        )

        self.pooler = BertPooler(config)
        self.model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base").to(DEVICE)

        # 🔥 获取消融实验变体配置
        self.ablation_variant = getattr(multimodal_config, 'ablation_variant', 'GICA')
        
        print(f"\n🔥 初始化消融实验变体: {self.ablation_variant}")
        
        # ITHP参数配置
        ITHP_args = {
            'X0_dim': TEXT_DIM,
            'X1_dim': ACOUSTIC_DIM,
            'X2_dim': VISUAL_DIM,
            'B0_dim': multimodal_config.B0_dim,
            'B1_dim': multimodal_config.B1_dim,
            'inter_dim': multimodal_config.inter_dim,
            'max_sen_len': multimodal_config.max_seq_length,
            'drop_prob': multimodal_config.drop_prob,
            'p_beta': multimodal_config.p_beta,
            'p_gamma': multimodal_config.p_gamma,
            'p_lambda': multimodal_config.p_lambda,
            'gating_mode': getattr(multimodal_config, 'gating_mode', 'dual_gating'),
        }

        # 🔥 根据变体决定是否需要ITHP模块
        if self.ablation_variant in ['T+V+A', 'GICA']:
            # 层次结构变体，需要完整的ITHP模块
            self.ITHP = ITHP(ITHP_args)
            self.expand = nn.Linear(multimodal_config.B1_dim, TEXT_DIM)
            print(f"   ✓ 使用ITHP层次结构")
        else:
            self.ITHP = None
            self.expand = None
        
        # 🔥 为声学特征添加投影层
        if self.ablation_variant in ['A', 'T+A', 'A+V', 'T+A+V', 'T+V+A', 'GICA']:
            self.acoustic_proj = nn.Linear(ACOUSTIC_DIM, TEXT_DIM)
            print(f"   ✓ 声学投影层 ({ACOUSTIC_DIM} → {TEXT_DIM})")
        else:
            self.acoustic_proj = None
            
        # 🔥 为视觉特征添加投影层
        if self.ablation_variant in ['V', 'T+V', 'A+V', 'T+A+V', 'T+V+A', 'GICA']:
            self.visual_proj = nn.Linear(VISUAL_DIM, TEXT_DIM)
            print(f"   ✓ 视觉投影层 ({VISUAL_DIM} → {TEXT_DIM})")
        else:
            self.visual_proj = None
        
        # 🔥 为单模态(A, V)添加MLP处理路径
        if self.ablation_variant in ['A', 'V']:
            self.single_modal_mlp = nn.Sequential(
                nn.Linear(TEXT_DIM, TEXT_DIM),
                nn.ReLU(),
                nn.Dropout(multimodal_config.dropout_prob),
                nn.Linear(TEXT_DIM, TEXT_DIM)
            )
            print(f"   ✓ 单模态MLP")
        else:
            self.single_modal_mlp = None
        
        # 🔥 为双模态Layer 1添加融合MLP (T+A, T+V, A+V)
        if self.ablation_variant in ['T+A', 'T+V', 'A+V']:
            self.dual_fusion_mlp = nn.Sequential(
                nn.Linear(TEXT_DIM * 2, TEXT_DIM),
                nn.ReLU(),
                nn.Dropout(multimodal_config.dropout_prob)
            )
            print(f"   ✓ 双模态融合MLP")
        else:
            self.dual_fusion_mlp = None
        
        # 🔥 为T+A+V (flat)添加三模态融合MLP
        if self.ablation_variant == 'T+A+V':
            self.triple_fusion_mlp = nn.Sequential(
                nn.Linear(TEXT_DIM * 3, TEXT_DIM),
                nn.ReLU(),
                nn.Dropout(multimodal_config.dropout_prob)
            )
            print(f"   ✓ 三模态融合MLP")
        else:
            self.triple_fusion_mlp = None
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(multimodal_config.dropout_prob)
        self.beta_shift = multimodal_config.beta_shift

        # 保留原有配置（用于GICA变体的内部控制）
        self.fusion_mode = getattr(multimodal_config, 'fusion_mode', 'full')
        self.gating_mode = getattr(multimodal_config, 'gating_mode', 'dual_gating')

        self.init_weights()
        print(f"   ✓ 模型初始化完成\n")

    def forward(self, input_ids, visual, acoustic, attention_mask=None, epoch=0, max_epochs=40):
        # 获取文本特征
        embedding_output = self.model(input_ids, attention_mask=attention_mask)
        x = embedding_output[0]  # [batch, seq_len, TEXT_DIM]
        
        # 初始化返回值
        IB_total = torch.tensor(0.0).to(DEVICE)
        kl_loss_0 = torch.tensor(0.0).to(DEVICE)
        mse_0 = torch.tensor(0.0).to(DEVICE)
        kl_loss_1 = torch.tensor(0.0).to(DEVICE)
        mse_1 = torch.tensor(0.0).to(DEVICE)
        
        # 🔥 根据变体执行不同的forward流程
        if self.ablation_variant == 'T':
            # 变体1: 仅文本
            acoustic_vis_embedding = torch.zeros_like(x)
            use_text_residual = True  # T变体使用文本
            
        elif self.ablation_variant == 'A':
            # 变体2: 仅声学 - 完全不使用文本和视觉
            acoustic_proj = self.acoustic_proj(acoustic)
            acoustic_processed = self.single_modal_mlp(acoustic_proj)
            acoustic_vis_embedding = self.beta_shift * acoustic_processed
            use_text_residual = False  # 🔥 A变体不使用文本残差
            
        elif self.ablation_variant == 'V':
            # 变体3: 仅视觉 - 完全不使用文本和声学
            visual_proj = self.visual_proj(visual)
            visual_processed = self.single_modal_mlp(visual_proj)
            acoustic_vis_embedding = self.beta_shift * visual_processed
            use_text_residual = False  # 🔥 V变体不使用文本残差
            
        elif self.ablation_variant == 'T+A':
            # 变体4: 双模态(文本+声学)
            acoustic_proj = self.acoustic_proj(acoustic)
            fused = torch.cat([x, acoustic_proj], dim=-1)
            fused_features = self.dual_fusion_mlp(fused)
            acoustic_vis_embedding = self.beta_shift * fused_features
            use_text_residual = True  # 双模态使用文本
            
        elif self.ablation_variant == 'T+V':
            # 变体5: 双模态(文本+视觉)
            visual_proj = self.visual_proj(visual)
            fused = torch.cat([x, visual_proj], dim=-1)
            fused_features = self.dual_fusion_mlp(fused)
            acoustic_vis_embedding = self.beta_shift * fused_features
            use_text_residual = True  # 双模态使用文本
            
        elif self.ablation_variant == 'A+V':
            # 变体6: 双模态(声学+视觉) - 完全不使用文本
            acoustic_proj = self.acoustic_proj(acoustic)
            visual_proj = self.visual_proj(visual)
            fused = torch.cat([acoustic_proj, visual_proj], dim=-1)
            fused_features = self.dual_fusion_mlp(fused)
            acoustic_vis_embedding = self.beta_shift * fused_features
            use_text_residual = False  # 🔥 A+V变体不使用文本残差
            
        elif self.ablation_variant == 'T+A+V':
            # 变体7: 三模态flat
            acoustic_proj = self.acoustic_proj(acoustic)
            visual_proj = self.visual_proj(visual)
            fused = torch.cat([x, acoustic_proj, visual_proj], dim=-1)
            fused_features = self.triple_fusion_mlp(fused)
            acoustic_vis_embedding = self.beta_shift * fused_features
            use_text_residual = True  # 三模态使用文本
            
        elif self.ablation_variant == 'T+V+A':
            # 变体8: 层次结构 - Layer1: T+V→B_G, Layer2: B_G+A→B_1
            b1, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, intermediate_results = self.ITHP(
                x, visual, acoustic, epoch, max_epochs
            )
            
            reconstructions = intermediate_results['reconstructions']
            h_m = self.expand(reconstructions['b1'])
            
            if self.fusion_mode == 'full':
                acoustic_recon = reconstructions['acoustic_recon']
                visual_recon = reconstructions['visual_recon']
                acoustic_proj = self.acoustic_proj(acoustic_recon)
                visual_proj = self.visual_proj(visual_recon)
                acoustic_vis_embedding = self.beta_shift * (h_m + acoustic_proj + visual_proj)
            else:
                acoustic_vis_embedding = self.beta_shift * h_m
            use_text_residual = True  # 层次结构使用文本
            
        elif self.ablation_variant == 'GICA':
            # 变体9: 完整GICA模型 - Layer1: T+A→B_G, Layer2: B_G+V→B_1
            b1, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, intermediate_results = self.ITHP(
                x, acoustic, visual, epoch, max_epochs
            )
            
            reconstructions = intermediate_results['reconstructions']
            h_m = self.expand(reconstructions['b1'])
            
            if self.fusion_mode == 'b1_only':
                acoustic_vis_embedding = self.beta_shift * h_m
            elif self.fusion_mode == 'b1_acoustic':
                acoustic_recon = reconstructions['acoustic_recon']
                acoustic_proj = self.acoustic_proj(acoustic_recon)
                acoustic_vis_embedding = self.beta_shift * (h_m + acoustic_proj)
            elif self.fusion_mode == 'b1_visual':
                visual_recon = reconstructions['visual_recon']
                visual_proj = self.visual_proj(visual_recon)
                acoustic_vis_embedding = self.beta_shift * (h_m + visual_proj)
            elif self.fusion_mode == 'full':
                acoustic_recon = reconstructions['acoustic_recon']
                visual_recon = reconstructions['visual_recon']
                acoustic_proj = self.acoustic_proj(acoustic_recon)
                visual_proj = self.visual_proj(visual_recon)
                acoustic_vis_embedding = self.beta_shift * (h_m + acoustic_proj + visual_proj)
            else:
                raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
            use_text_residual = True  # GICA使用文本
        
        else:
            raise ValueError(f"Unknown ablation_variant: {self.ablation_variant}")

        # 🔥 最终输出 - 根据变体决定是否加文本残差
        if use_text_residual:
            # 包含文本的变体：T, T+A, T+V, T+A+V, T+V+A, GICA
            sequence_output = self.dropout(self.LayerNorm(acoustic_vis_embedding + x))
        else:
            # 纯单模态(A, V)和A+V变体：不加文本残差
            sequence_output = self.dropout(self.LayerNorm(acoustic_vis_embedding))
        
        pooled_output = self.pooler(sequence_output)

        return pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1


class ITHP_DeBertaForSequenceClassification(DebertaV2PreTrainedModel):
    """支持消融实验的分类模型"""
    
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dberta = ITHP_DebertaModel(config, multimodal_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, visual, acoustic, attention_mask=None, epoch=0, max_epochs=40):
        pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1 = self.dberta(
            input_ids, visual, acoustic, attention_mask=attention_mask, epoch=epoch, max_epochs=max_epochs
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1
