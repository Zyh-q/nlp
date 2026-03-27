from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


input = 'I like you.'
semantic_cls = pipeline(Tasks.text_classification, 'iic/nlp_bert_sentiment-analysis_english-base')
result = semantic_cls(input)

print('输入文本:\n{}\n'.format(input))
print('分类结果:\n{}'.format(result))