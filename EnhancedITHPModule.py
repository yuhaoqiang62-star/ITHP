import torch
import torch.nn as nn
import torch.nn.functional as F
from ITHP import ITHP


class AdversarialAlignmentModule(nn.Module):
    """基于AGF-IB模型设计的对抗性对齐模块"""

    def __init__(self, hidden_dim=128):
        super(AdversarialAlignmentModule, self).__init__()

        # 生成器：将不同模态映射到共同特征空间
        self.generator_text = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.generator_audio = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.generator_visual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 判别器：区分特征来源模态
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 3),  # 3个模态
            nn.Softmax(dim=-1)
        )

        self.adversarial_loss = nn.CrossEntropyLoss()

    def forward(self, text_feat, audio_feat, visual_feat):
        # 生成器映射
        text_aligned = self.generator_text(text_feat)
        audio_aligned = self.generator_audio(audio_feat)
        visual_aligned = self.generator_visual(visual_feat)

        # 判别器预测
        text_pred = self.discriminator(text_aligned.detach())
        audio_pred = self.discriminator(audio_aligned.detach())
        visual_pred = self.discriminator(visual_aligned.detach())

        # 真实标签
        batch_size = text_feat.size(0)
        text_labels = torch.zeros(batch_size).long().to(text_feat.device)
        audio_labels = torch.ones(batch_size).long().to(text_feat.device)
        visual_labels = torch.full((batch_size,), 2).long().to(text_feat.device)

        # 对抗损失
        d_loss = (self.adversarial_loss(text_pred, text_labels) +
                  self.adversarial_loss(audio_pred, audio_labels) +
                  self.adversarial_loss(visual_pred, visual_labels)) / 3

        # 生成器损失（希望判别器无法区分）
        fake_labels = torch.randint(0, 3, (batch_size,)).to(text_feat.device)
        g_loss = (self.adversarial_loss(self.discriminator(text_aligned), fake_labels) +
                  self.adversarial_loss(self.discriminator(audio_aligned), fake_labels) +
                  self.adversarial_loss(self.discriminator(visual_aligned), fake_labels)) / 3

        return text_aligned, audio_aligned, visual_aligned, d_loss, g_loss


class MultiModalFusionGate(nn.Module):
    """基于AGF-IB的多模态融合门控机制"""

    def __init__(self, hidden_dim):
        super(MultiModalFusionGate, self).__init__()

        # 模态间注意力机制
        self.text_audio_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.text_visual_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.audio_visual_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, text_feat, audio_feat, visual_feat):
        # 模态间门控融合
        ta_concat = torch.cat([text_feat, audio_feat], dim=-1)
        ta_weight = self.text_audio_gate(ta_concat)
        ta_fused = ta_weight * text_feat + (1 - ta_weight) * audio_feat

        tv_concat = torch.cat([text_feat, visual_feat], dim=-1)
        tv_weight = self.text_visual_gate(tv_concat)
        tv_fused = tv_weight * text_feat + (1 - tv_weight) * visual_feat

        av_concat = torch.cat([audio_feat, visual_feat], dim=-1)
        av_weight = self.audio_visual_gate(av_concat)
        av_fused = av_weight * audio_feat + (1 - av_weight) * visual_feat

        # 三模态融合
        final_fused = (ta_fused + tv_fused + av_fused) / 3

        return final_fused


class EnhancedITHPModule(nn.Module):
    """增强的ITHP模块，集成对抗性对齐和高级融合机制"""

    def __init__(self, ITHP_args):
        super(EnhancedITHPModule, self).__init__()

        # 保持原始ITHP不变
        self.ithp = ITHP(ITHP_args)

        # 新增组件
        self.adversarial_alignment = AdversarialAlignmentModule(ITHP_args['B1_dim'])
        self.fusion_gate = MultiModalFusionGate(ITHP_args['B1_dim'])

        # 特征增强网络
        self.feature_enhancer = nn.Sequential(
            nn.Linear(ITHP_args['B1_dim'], ITHP_args['B1_dim'] * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(ITHP_args['B1_dim'] * 2, ITHP_args['B1_dim'])
        )

        # 信息瓶颈正则化器
        self.ib_regularizer = nn.Sequential(
            nn.Linear(ITHP_args['B1_dim'], ITHP_args['B1_dim'] // 2),
            nn.ReLU(),
            nn.Linear(ITHP_args['B1_dim'] // 2, ITHP_args['B1_dim'])
        )

        # 跨模态一致性损失
        self.consistency_loss = nn.MSELoss()

    def compute_mutual_information_loss(self, z, x):
        """计算互信息损失（基于KL散度近似）"""
        # 计算z的边缘分布
        z_marginal = torch.mean(z, dim=0, keepdim=True)
        z_marginal = z_marginal.repeat(z.size(0), 1)

        # KL散度作为互信息的下界
        kl_div = F.kl_div(F.log_softmax(z, dim=-1),
                          F.softmax(z_marginal, dim=-1),
                          reduction='batchmean')

        return kl_div

    def forward(self, x, visual, acoustic):
        # 1. 原始ITHP处理
        b1_original, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1 = self.ithp(x, visual, acoustic)

        # 2. 特征增强
        b1_enhanced = self.feature_enhancer(b1_original)

        # 3. 对抗性对齐（假设我们可以分解b1为不同模态的表示）
        # 这里我们创建伪模态特征进行演示
        text_feat = b1_enhanced
        audio_feat = self.ib_regularizer(b1_enhanced)  # 变换得到不同视角
        visual_feat = F.dropout(b1_enhanced, p=0.1, training=self.training)  # 另一种变换

        text_aligned, audio_aligned, visual_aligned, d_loss, g_loss = \
            self.adversarial_alignment(text_feat, audio_feat, visual_feat)

        # 4. 多模态融合
        fused_features = self.fusion_gate(text_aligned, audio_aligned, visual_aligned)

        # 5. 计算额外的信息瓶颈损失
        mi_loss = self.compute_mutual_information_loss(fused_features, b1_original)

        # 6. 一致性损失
        consistency_loss = (self.consistency_loss(text_aligned, audio_aligned) +
                            self.consistency_loss(text_aligned, visual_aligned) +
                            self.consistency_loss(audio_aligned, visual_aligned)) / 3

        # 7. 总损失
        enhanced_IB_loss = IB_total + 0.1 * (d_loss + g_loss) + 0.05 * mi_loss + 0.02 * consistency_loss

        return fused_features, enhanced_IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1, d_loss, g_loss, mi_loss, consistency_loss


# 使用示例
def create_enhanced_ithp(ITHP_args):
    """创建增强版ITHP模块的工厂函数"""
    return EnhancedITHPModule(ITHP_args)