import torch
from torch.utils.data import Dataset

class PromptDataset(Dataset):
    """
    一个简单的数据集，用于在多个GPU上生成类别图像的提示词。
    """

    def __init__(self, prompt, num_samples):
        """
        初始化PromptDataset
        
        Args:
            prompt (str): 要使用的提示词
            num_samples (int): 样本数量
        """
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    """
    使用tokenizer对提示词进行编码
    
    Args:
        tokenizer: 用于编码的tokenizer
        prompt (str): 要编码的提示词
        max_sequence_length (int): 最大序列长度
        
    Returns:
        torch.Tensor: 编码后的输入ID
    """
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# 测试代码
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # 测试PromptDataset
    prompt = "a photo of a dog"
    num_samples = 5
    
    dataset = PromptDataset(prompt, num_samples)
    print(f"数据集大小: {len(dataset)}")
    
    # 测试获取样本
    for i in range(3):  # 只打印前三个样本
        sample = dataset[i]
        print(f"样本 {i}: {sample}")
    
    # 测试tokenize_prompt函数
    # 加载一个tokenizer进行测试
    try:
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # 测试编码
        max_length = 77  # CLIP的默认长度
        encoded = tokenize_prompt(tokenizer, prompt, max_length)
        
        print(f"编码形状: {encoded.shape}")
        print(f"编码内容: {encoded}")
        
        # 解码回原始文本
        decoded = tokenizer.decode(encoded[0])
        print(f"解码结果: {decoded}")
    except Exception as e:
        print(f"Tokenizer测试失败: {e}")
        print("跳过tokenizer测试，可能需要安装transformers库")
