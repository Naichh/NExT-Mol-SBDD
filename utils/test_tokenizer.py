from transformers import AutoTokenizer

def inspect_the_tokenizer():
    # --- 1. 加载预训练好的Tokenizer ---
    # 这行代码会从Hugging Face下载并加载与MoLlama模型配套的分词器
    # 这完全模拟了你提供的init_tokenizer函数中的核心步骤
    model_name = "acharkq/MoLlama"
    print(f"--- 正在加载模型 '{model_name}' 的Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer加载成功！\n")

    # --- 2. 查看Tokenizer的“密码本”长什么样 ---
    print("--- Tokenizer的'密码本'（词汇表）里有什么？ ---")

    # 获取完整的词汇表，它是一个 单词 -> 编号 的字典
    vocabulary = tokenizer.get_vocab()
    print(f"词汇表总大小: {len(vocabulary)} 个'单词'\n")

    print("我们来看一下词汇表里前20个'单词'和它们的编号 (ID):")
    # items()[:20] 这种写法不安全，我们换一种方式
    count = 0
    for token, token_id in vocabulary.items():
        if count < 20:
            print(f"  - 单词: '{token}',  编号: {token_id}")
            count += 1
        else:
            break
    print("...\n")

    # --- 3. 完整演示一次编码和解码的过程 ---
    print("--- 演示对一个分子进行编码和解码 ---")
    # 假设我们有一个丙酮分子，它的SELFIES字符串是 '[C][C][=Branch1][C][=O]'
    selfies_string = '[C][C][=Branch1][C][=O]'
    print(f"原始SELFIES字符串: {selfies_string}\n")

    # a. 分词 (Tokenization) - 切分成“单词”
    tokens = tokenizer.tokenize(selfies_string)
    print(f"1. 分词后的结果 (Tokens): {tokens}")
    print("   (Tokenizer自动把字符串切分成了它认识的化学片段)\n")

    # b. 编码 (Encoding) - 文本“单词”转换为数字ID
    encoded_ids = tokenizer.encode(selfies_string)
    print(f"2. 编码后的数字ID: {encoded_ids}")
    print("   (每个Token都被转换成了它在词汇表里的唯一编号)\n")

    # c. 解码 (Decoding) - 数字ID转换回文本字符串
    decoded_string = tokenizer.decode(encoded_ids)
    print(f"3. 将数字ID解码回字符串: {decoded_string}")
    print("   (解码过程会把特殊符号比如<s>和</s>也显示出来)\n")

    # --- 4. 解释特殊符号 ---
    print("--- 关于特殊符号 ---")
    print(f"BOS (Beginning of Sequence) Token: '{tokenizer.bos_token}', ID: {tokenizer.bos_token_id}")
    print(f"EOS (End of Sequence) Token: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
    print("BOS和EOS就像句子的开头和句号，用来告诉模型一个序列从哪里开始，到哪里结束。")
    print(f"这就是为什么编码后的ID列表开头是 {tokenizer.bos_token_id}，结尾是 {tokenizer.eos_token_id}。")

if __name__ == '__main__':
    inspect_the_tokenizer()
