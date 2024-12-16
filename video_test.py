import google.generativeai as genai
import time

genai.configure(api_key='AIzaSyBZpCdUCTqvrBiCc4ntQQFxrOmwqqAwVH4')

video_file_name = "/pub0/oclayton/reflect_multimodal/main/thor_tasks/makeSalad/makeSalad-5/original-video.mp4"

print(f"Uploading file...")
video_file = genai.upload_file(path=video_file_name)
print(f"Completed upload: {video_file.uri}")



# Check whether the file is ready to be used.
while video_file.state.name == "PROCESSING":
    print('.', end='')
    time.sleep(10)
    video_file = genai.get_file(video_file.name)

if video_file.state.name == "FAILED":
  raise ValueError(video_file.state.name)

# Create the prompt.
prompt = "Summarize this video."

# Choose a Gemini model.
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Make the LLM request.
print("Making LLM inference request...")
response = model.generate_content([video_file, prompt])

# Print the response, rendering any Markdown
print(response.text)
print(video_file.name)