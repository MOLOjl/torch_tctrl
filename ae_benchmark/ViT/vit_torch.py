import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

USE_DTR = True if os.environ.get('DTR_ENABLE') == '1' else False
RECORD_MEM_SNAPSHOT = True if os.environ.get('RECORD_MEM_SNAPSHOT') == '1' else False
COST_FIRST_EVICT = True if os.environ.get('COST_FIRST_EVICT') == '1' else False
ORIG_DTR = True if os.environ.get('ORIG_DTR') == '1' else False
snapshot_filename = os.environ.get('SNAP_FILE_NAME')
mem_budget = float(os.environ.get('MEM_BUDGET', 0))

# 定义 Vision Transformer 模型
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.size()

        # 计算 Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 计算注意力输出
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 线性变换
        output = self.out_proj(attn_output)
        return output


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, num_classes=10, embed_dim=768, num_heads=12,
                 num_layers=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        encoder_layer = CustomTransformerEncoderLayer(embed_dim, num_heads)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer_encoder(x)
        cls_token_output = x[:, 0]
        x = self.head(cls_token_output)
        return x

# 超参数设置
image_size = 224
patch_size = 16
in_channels = 3
num_classes = 10
embed_dim = 1024
num_heads = 16
num_layers = 32
batch_size = 128
learning_rate = 0.001

def main(num_epochs, model, criterion, optimizer):
    # 生成 mock 数据
    mock_images = torch.randn(batch_size, in_channels, image_size, image_size).to('cuda', dtype=torch.bfloat16)
    mock_labels = torch.randint(0, num_classes, (batch_size,)).to('cuda')
    if ORIG_DTR:
        mock_images = mock_images.checkpoint()
        mock_labels = mock_labels.checkpoint()
    # 训练循环
    for epoch in range(num_epochs):
        # optimizer.zero_grad()
        outputs = model(mock_images)
        loss = criterion(outputs, mock_labels)
        loss.backward()
        # optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    


if __name__ == '__main__':
    if USE_DTR:
        torch.init_dtb_manager()
        if mem_budget > 0:
            torch.set_memory_budget(int(mem_budget * 1e10))
            print(f"Set memory budget to {mem_budget} GB")

     # 创建模型
    model = ViT(image_size, patch_size, in_channels, num_classes, embed_dim, num_heads, num_layers)
    # 将模型移到CUDA设备上
    model = model.to('cuda', dtype=torch.bfloat16).train()

    if USE_DTR:
        model._apply(lambda v: v.detach().checkpoint(True))

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 记录开始时间
    num_epochs = 40
    start_time = time.time()

    # 记录开始时的内存信息
    # torch.cuda.reset_peak_memory_stats()
    # torch.cuda.empty_cache()
    if ORIG_DTR:
        for _ in range(num_epochs):
            main(1, model, criterion, optimizer)
    else:
        main(num_epochs, model, criterion, optimizer)

    # 记录结束时间
    torch.cuda.synchronize()
    end_time = time.time()
    # 计算训练耗时
    elapsed_time = end_time - start_time

    # 统计内存使用情况
    allocated_memory = torch.cuda.max_memory_allocated()
    peak_memory = torch.cuda.max_memory_reserved()
    fragmentation = (peak_memory-allocated_memory) / peak_memory

    print(f'Training throughput: {(elapsed_time / num_epochs):.2f} seconds.')
    print(f'Allocated Memory: {allocated_memory / (1024 ** 2):.2f} MB')
    print(f'Peak Memory: {peak_memory / (1024 ** 2):.2f} MB')
    print(f'Memory Fragmentation: {fragmentation}')    
    print(f'[Summary] {allocated_memory / (1024 ** 2)} {peak_memory / (1024 ** 2)} {fragmentation} {(elapsed_time / num_epochs)*1000}')