import os
import time
import google.generativeai as genai
import openai
import os
import json
import datetime
import numpy as np

class LLMPrompter():
    def __init__(self, gpt_version, api_key_str) -> None:
        self.gpt_version = gpt_version
        if api_key_str is None:
            raise ValueError("OpenAI API key is not provided.")
        else:
            genai.configure(api_key=api_key_str)

    def query(self, prompt: str, sampling_params: dict, save: bool, save_dir: str) -> str:
        while True:
            try:
                if 'gemini-1.5-flash' in self.gpt_version:
                    print(prompt['system'])
                    print(prompt['user'])
                    model = genai.GenerativeModel(self.gpt_version, 
                                                  system_instruction = prompt['system'])
                    response = model.generate_content(prompt['user'])
                    print(response)
                else:
                    response = openai.Completion.create(
                        model=self.gpt_version,
                        prompt=prompt,
                        **sampling_params
                    )
            except Exception as e:
                print("Request failed, sleep 2 secs and try again...", e)
                time.sleep(2)
                continue
            break

        if save:
            key = self.make_key()
            output = {}
            os.system('mkdir -p {}'.format(save_dir))
            if os.path.exists(os.path.join(save_dir, 'response.json')):
                with open(os.path.join(save_dir, 'response.json'), 'r') as f:
                    prev_response = json.load(f)
                    output = prev_response

            with open(os.path.join(save_dir, 'response.json'), 'w') as f:
                if 'gemini-1.5-flash' in self.gpt_version:
                    output[key] = {
                                'prompt': prompt,
                                'sampling_params': sampling_params,
                                'response': response._result.candidates[0].content.parts[0].text.strip()

                            }
                else:
                    output[key] = {
                                'prompt': prompt,
                                'sampling_params': sampling_params,
                                'response': response['choices'][0]['text'].strip(),
                                'logprob': np.mean(response['choices'][0]['logprobs']['token_logprobs'])
                            }
                json.dump(output, f, indent=4)
            
        if 'gemini-1.5-flash' in self.gpt_version:
            return response._result.candidates[0].content.parts[0].text.strip(), None
        else:
            return response['choices'][0]['text'].strip(), np.mean(response['choices'][0]['logprobs']['token_logprobs'])

    def make_key(self):
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")