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
    
    def upload_video(self, videoPath) -> str:
        try: 
            video_file = genai.upload_file(path = videoPath)
            while video_file.state.name == "PROCESSING":
                time.sleep(5)
                video_file = genai.get_file(video_file.name)
            if(video_file.state.name == "FAILED"):
                raise ValueError(video_file.state.name)
        except Exception as e:
                print("Video Upload Failed... ", e)
        return(video_file.name)
    

    def query(self, prompt: str, sampling_params: dict, save: bool, save_dir: str, videoName = '') -> str:
        while True:
            try:
                if 'gemini-1.5-flash' in self.gpt_version:
                    print(prompt['system'])
                    print(prompt['user'])
                    if (videoName == ''):
                        model = genai.GenerativeModel(self.gpt_version, 
                                                    system_instruction = prompt['system'])
                        response = model.generate_content(prompt['user'], 
                                                        generation_config=genai.types.GenerationConfig(**sampling_params))
                    else: 
                        video_file = genai.get_file(videoName)
                        model = genai.GenerativeModel(self.gpt_version, 
                                                    system_instruction = prompt['system'])
                        response = model.generate_content([video_file, prompt['user']], generation_config=genai.types.GenerationConfig(**sampling_params))
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