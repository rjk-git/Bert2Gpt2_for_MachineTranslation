# BERT2GPT2

### <u>描述</u>

- 基于huggingface的[Transformers](https://github.com/huggingface/transformers)库实现。在编码器端与解码器端分别借助BERT和GPT2预训练成果，以期待模型能够更好的热启动，在使用小规模平行语料进行机器翻译训练任务中能够提供帮助。同时借助GPT2的单向语言模型能力，使生成句子更加顺畅。

### <u>支持</u>

- [huggingface/models](https://huggingface.co/models)中支持的BERT模型和GPT2模型作为源语言，目标语言的编码器解码器。

### <u>使用</u>

1. 准备训练文件和验证文件，每行为一对平行语料，默认以“\t”分隔。
2. 参考run_example.sh设置训练文件路径、模型保存路径、训练参数等。