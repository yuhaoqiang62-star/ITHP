import torch
import torch.nn as nn
import torch.nn.functional as F
import global_configs
import math


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim=256, num_heads=8):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # 投影层
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, query_dim)

        # 层归一化和dropout
        self.layer_norm = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(0.1)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.shape[:2]

        # 线性投影
        Q = self.query_proj(query)  # [batch_size, seq_len, hidden_dim]
        K = self.key_proj(key)  # [batch_size, seq_len, hidden_dim]
        V = self.value_proj(value)  # [batch_size, seq_len, hidden_dim]

        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        attended_values = torch.matmul(attention_weights, V)

        # 重塑回原始维度
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )

        # 输出投影
        output = self.output_proj(attended_values)

        # 残差连接和层归一化
        output = self.layer_norm(output + query)

        return output, attention_weights.mean(dim=1)  # 返回平均注意力权重用于可视化


class LearnableWeights(nn.Module):
    """可学习的损失权重参数"""

    def __init__(self, initial_beta=8.0, initial_gamma=32.0, initial_lambda=0.3):
        super(LearnableWeights, self).__init__()

        # 使用log参数确保权重始终为正
        self.log_beta = nn.Parameter(torch.log(torch.tensor(initial_beta, dtype=torch.float32)))
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(initial_gamma, dtype=torch.float32)))
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(initial_lambda, dtype=torch.float32)))

        # 温度参数用于控制学习速度
        self.temperature = nn.Parameter(torch.tensor(1.0))

    @property
    def beta(self):
        return torch.exp(self.log_beta / self.temperature)

    @property
    def gamma(self):
        return torch.exp(self.log_gamma / self.temperature)

    @property
    def lambda_weight(self):
        return torch.exp(self.log_lambda / self.temperature)

    def get_weights(self):
        """返回当前权重值，用于监控"""
        return {
            'beta': self.beta.item(),
            'gamma': self.gamma.item(),
            'lambda': self.lambda_weight.item(),
            'temperature': self.temperature.item()
        }


class GatedBottleneck(nn.Module):
    """门控瓶颈机制"""

    def __init__(self, input_dim, bottleneck_dim, gate_activation='sigmoid'):
        super(GatedBottleneck, self).__init__()

        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.Sigmoid() if gate_activation == 'sigmoid' else nn.Tanh()
        )

        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim * 2),  # 输出mu和logvar
        )

        # 重要性门控（用于特征选择）
        self.importance_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # 输入特征重要性门控
        importance_weights = self.importance_gate(x)
        gated_input = x * importance_weights
        gated_input = self.layer_norm(gated_input + x)  # 残差连接

        # 计算门控值
        gate_values = self.gate_network(gated_input)

        # 编码
        h = self.encoder(gated_input)
        mu, logvar = h.chunk(2, dim=-1)

        # 应用门控到均值
        gated_mu = mu * gate_values

        # 门控也影响方差，但程度较小
        gate_logvar_effect = torch.log(gate_values + 1e-8) * 0.1
        gated_logvar = logvar + gate_logvar_effect

        return gated_mu, gated_logvar, gate_values, importance_weights


class ITHP(nn.Module):
    """改进的ITHP模型"""

    def __init__(self, ITHP_args):
        super(ITHP, self).__init__()

        # 原始参数
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = (global_configs.TEXT_DIM, global_configs.ACOUSTIC_DIM,
                                              global_configs.VISUAL_DIM)

        self.X0_dim = ITHP_args['X0_dim']
        self.X1_dim = ITHP_args['X1_dim']
        self.X2_dim = ITHP_args['X2_dim']
        self.inter_dim = ITHP_args['inter_dim']
        self.drop_prob = ITHP_args['drop_prob']
        self.max_sen_len = ITHP_args['max_sen_len']
        self.B0_dim = ITHP_args['B0_dim']
        self.B1_dim = ITHP_args['B1_dim']

        # 可学习权重
        self.learnable_weights = LearnableWeights(
            initial_beta=ITHP_args.get('p_beta', 8.0),
            initial_gamma=ITHP_args.get('p_gamma', 32.0),
            initial_lambda=ITHP_args.get('p_lambda', 0.3)
        )

        # 第一层门控瓶颈编码器
        self.gated_encoder1 = GatedBottleneck(
            self.X0_dim, self.B0_dim, gate_activation='sigmoid'
        )

        # 跨模态注意力（文本-声学）
        self.text_acoustic_attention = CrossModalAttention(
            query_dim=self.B0_dim,
            key_dim=self.X1_dim,
            value_dim=self.X1_dim,
            hidden_dim=min(256, self.B0_dim),
            num_heads=8
        )

        # 第一层MLP（重构声学特征）
        self.MLP1 = nn.Sequential(
            nn.Linear(self.B0_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.X1_dim),
            nn.Sigmoid(),
            nn.Dropout(self.drop_prob),
        )

        # 第二层门控瓶颈编码器
        self.gated_encoder2 = GatedBottleneck(
            self.B0_dim, self.B1_dim, gate_activation='sigmoid'
        )

        # 跨模态注意力（B0-视觉）
        self.b0_visual_attention = CrossModalAttention(
            query_dim=self.B1_dim,
            key_dim=self.X2_dim,
            value_dim=self.X2_dim,
            hidden_dim=min(256, self.B1_dim),
            num_heads=8
        )

        # 第二层MLP（重构视觉特征）
        self.MLP2 = nn.Sequential(
            nn.Linear(self.B1_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.X2_dim),
            nn.Sigmoid(),
            nn.Dropout(self.drop_prob),
        )

        # 额外的融合层
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(self.B1_dim + self.X1_dim + self.X2_dim, self.B1_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.B1_dim, self.B1_dim),
        )

        self.criterion = nn.MSELoss()

        # 用于存储注意力权重（便于可视化）
        self.attention_weights = {}

    def kl_loss(self, mu, logvar):
        """计算KL散度损失"""
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_mean = torch.mean(kl_div)
        return kl_mean

    def reparameterise(self, mu, logvar):
        """重参数化技巧"""
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def adaptive_weight_decay(self, epoch, max_epochs):
        """自适应权重衰减"""
        progress = epoch / max_epochs
        decay_factor = 1.0 - 0.5 * progress  # 随训练进程减少权重影响
        return decay_factor

    def forward(self, x, visual, acoustic, epoch=0, max_epochs=100):
        # 获取当前可学习权重
        current_weights = self.learnable_weights.get_weights()
        beta = self.learnable_weights.beta
        gamma = self.learnable_weights.gamma
        lambda_weight = self.learnable_weights.lambda_weight

        # 自适应权重衰减
        decay_factor = self.adaptive_weight_decay(epoch, max_epochs)

        # ================== 第一层处理 ==================
        # 门控瓶颈编码
        mu1, logvar1, gate1_values, importance1_weights = self.gated_encoder1(x)
        kl_loss_0 = self.kl_loss(mu1, logvar1)

        # 重参数化
        b0 = self.reparameterise(mu1, logvar1)

        # 跨模态注意力（B0特征关注声学特征）
        attended_b0, attention_weights_1 = self.text_acoustic_attention(
            query=b0, key=acoustic, value=acoustic
        )
        self.attention_weights['text_acoustic'] = attention_weights_1

        # 结合原始和注意力增强的特征
        enhanced_b0 = b0 + 0.3 * attended_b0  # 残差连接

        # 重构声学特征
        output1 = self.MLP1(enhanced_b0)
        mse_0 = self.criterion(output1, acoustic)

        # 第一层信息瓶颈损失
        IB0 = kl_loss_0 + beta * mse_0 * decay_factor

        # ================== 第二层处理 ==================
        # 使用增强的b0特征进行第二层编码
        mu2, logvar2, gate2_values, importance2_weights = self.gated_encoder2(enhanced_b0)
        kl_loss_1 = self.kl_loss(mu2, logvar2)

        # 重参数化
        b1 = self.reparameterise(mu2, logvar2)

        # 跨模态注意力（B1特征关注视觉特征）
        attended_b1, attention_weights_2 = self.b0_visual_attention(
            query=b1, key=visual, value=visual
        )
        self.attention_weights['b0_visual'] = attention_weights_2

        # 结合原始和注意力增强的特征
        enhanced_b1 = b1 + 0.3 * attended_b1  # 残差连接

        # 重构视觉特征
        output2 = self.MLP2(enhanced_b1)
        mse_1 = self.criterion(output2, visual)

        # 第二层信息瓶颈损失
        IB1 = kl_loss_1 + gamma * mse_1 * decay_factor

        # ================== 多模态融合 ==================
        # 将不同层的特征进行融合
        fusion_input = torch.cat([enhanced_b1, output1, output2], dim=-1)
        final_b1 = self.multimodal_fusion(fusion_input)

        # 总的信息瓶颈损失
        IB_total = IB0 + lambda_weight * IB1

        # 返回结果和中间变量（用于分析）
        intermediate_results = {
            'gate1_values': gate1_values,
            'gate2_values': gate2_values,
            'importance1_weights': importance1_weights,
            'importance2_weights': importance2_weights,
            'attention_weights': self.attention_weights.copy(),
            'current_weights': current_weights,
            'enhanced_b0': enhanced_b0,
            'enhanced_b1': enhanced_b1,
            'reconstructed_acoustic': output1,
            'reconstructed_visual': output2
        }

        return final_b1, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, intermediate_results


# ================== 使用示例和训练相关代码 ==================

class ImprovedTrainingLoop:
    """改进的训练循环，包含监控和可视化功能"""

    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # 监控指标
        self.training_stats = {
            'weights_history': [],
            'attention_stats': [],
            'gate_stats': []
        }

    def train_epoch_improved(self, train_dataloader, epoch, max_epochs):
        """改进的训练epoch"""
        self.model.train()
        epoch_stats = {
            'total_loss': 0.0,
            'ib_loss': 0.0,
            'kl_losses': [],
            'mse_losses': [],
            'weight_changes': {},
            'attention_entropy': []
        }

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, visual, acoustic, label_ids = batch

            # 数据预处理
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            visual_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

            # 前向传播
            logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1, intermediate = self.model(
                input_ids, visual_norm, acoustic_norm, epoch, max_epochs
            )

            # 主任务损失
            loss_fct = nn.MSELoss()
            main_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            # 总损失
            total_loss = main_loss + 0.1 * IB_loss  # 可调整IB损失权重

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            # 统计信息收集
            epoch_stats['total_loss'] += total_loss.item()
            epoch_stats['ib_loss'] += IB_loss.item()
            epoch_stats['kl_losses'].append([kl_loss_0.item(), kl_loss_1.item()])
            epoch_stats['mse_losses'].append([mse_0.item(), mse_1.item()])

            # 收集权重变化信息
            current_weights = intermediate['current_weights']
            epoch_stats['weight_changes'] = current_weights

            # 计算注意力熵（衡量注意力分布集中程度）
            for key, attention_weights in intermediate['attention_weights'].items():
                entropy = self.calculate_attention_entropy(attention_weights)
                epoch_stats['attention_entropy'].append(entropy.item())

        # 计算平均统计
        num_batches = len(train_dataloader)
        epoch_stats['total_loss'] /= num_batches
        epoch_stats['ib_loss'] /= num_batches
        epoch_stats['avg_attention_entropy'] = sum(epoch_stats['attention_entropy']) / len(
            epoch_stats['attention_entropy'])

        # 保存训练统计
        self.training_stats['weights_history'].append(epoch_stats['weight_changes'])
        self.training_stats['attention_stats'].append(epoch_stats['avg_attention_entropy'])

        return epoch_stats

    def calculate_attention_entropy(self, attention_weights):
        """计算注意力权重的熵"""
        # attention_weights: [batch_size, seq_len, seq_len]
        # 计算每个样本的平均注意力熵
        batch_size = attention_weights.shape[0]
        entropies = []

        for i in range(batch_size):
            attn = attention_weights[i].mean(dim=0)  # 平均所有head
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)  # 归一化
            entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1).mean()
            entropies.append(entropy)

        return torch.stack(entropies).mean()

    def print_training_stats(self, epoch, epoch_stats):
        """打印训练统计信息"""
        print(f"\n=== Epoch {epoch + 1} Stats ===")
        print(f"Total Loss: {epoch_stats['total_loss']:.4f}")
        print(f"IB Loss: {epoch_stats['ib_loss']:.4f}")
        print(f"Current Weights - Beta: {epoch_stats['weight_changes']['beta']:.2f}, "
              f"Gamma: {epoch_stats['weight_changes']['gamma']:.2f}, "
              f"Lambda: {epoch_stats['weight_changes']['lambda']:.2f}")
        print(f"Average Attention Entropy: {epoch_stats['avg_attention_entropy']:.4f}")
        print(f"KL Losses: Layer1={epoch_stats['kl_losses'][-1][0]:.4f}, "
              f"Layer2={epoch_stats['kl_losses'][-1][1]:.4f}")
        print(f"MSE Losses: Acoustic={epoch_stats['mse_losses'][-1][0]:.4f}, "
              f"Visual={epoch_stats['mse_losses'][-1][1]:.4f}")


# ================== 配置和使用示例 ==================

def create_improved_model(original_args):
    """创建改进的ITHP模型"""

    # 扩展原始参数
    improved_args = original_args.copy()

    # 添加新的超参数
    improved_args.update({
        'use_attention': True,
        'use_gating': True,
        'use_learnable_weights': True,
        'attention_heads': 8,
        'gate_activation': 'sigmoid',
        'fusion_dropout': 0.1,
        'weight_decay_schedule': True
    })

    return ITHP(improved_args)


if __name__ == "__main__":
    # 假设你在 global_configs 里有对应的维度定义
    ITHP_args = {
        'X0_dim': global_configs.TEXT_DIM,
        'X1_dim': global_configs.ACOUSTIC_DIM,
        'X2_dim': global_configs.VISUAL_DIM,
        'inter_dim': 128,
        'drop_prob': 0.1,
        'max_sen_len': 50,
        'B0_dim': 64,
        'B1_dim': 64,
        'p_beta': 1.0,
        'p_gamma': 1.0,
        'p_lambda': 1.0,
    }

    model = ITHP(ITHP_args)
    print(model)   # 打印网络结构
