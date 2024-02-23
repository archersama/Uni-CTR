#
#
#
# def rela_impr(auc_measured_model, auc_base_model):
#     return ((auc_measured_model - 0.5) / (auc_base_model - 0.5) - 1) * 100
#
# # Example usage of the function:
# auc_measured_model_example = 0.7391  # Example AUC for the measured model
# auc_base_model_example = 0.6979      # Example AUC for the base model
#
# # Calculate the Relative Improvement for the example:
# relative_improvement_example = rela_impr(auc_measured_model_example, auc_base_model_example)
#
# print("Relative Improvement: {:.2f}%".format(relative_improvement_example))


# """
# openai 数据接口
# 建议使用gpt-3.5-turbo
# """
#
# import openai
#
#
# class gptApi():
#
#     def __init__(self,model= "gpt-3.5-turbo",max_tokens=1024):
#         self.key = "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"
#         self.base_url = "https://api.openai.com"
#         self.model= model
#         # self.model= "gpt-3.5-turbo-16k"
#         # self.model= "gpt-4-1106-preview" #较低
#         # self.model= "gpt-4"
#         self.max_tokens = max_tokens
#
#     #生成
#     def generate(self,query):
#         openai.api_key = self.key
#         openai.api_base = self.base_url
#         messages = []
#         user_msg = {'role': 'user', 'content': f'''{query}'''}
#         messages.append(user_msg)
#         completion = openai.ChatCompletion.create(model=self.model,
#                                                   temperature=1.0,
#                                                   top_p=1.0,
#                                                   messages=messages,
#                                                   max_tokens=self.max_tokens)
#         answer = {}
#         answer['context'] = query
#         answer['response'] = (
#             completion['choices'][0]['message']['content']).strip()
#
#         return answer
#
#
#
#
# print(gptApi().generate("滚吧，zei了"))


