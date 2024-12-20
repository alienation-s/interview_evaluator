import requests
import json
# 测试转写接口
def test_transcribe():
    # files = {'file': open('/home/ifly/cs/interview_evaluator/silence.mp3', 'rb')}
    files = {'file': open('/home/ifly/cs/interview_evaluator/experiment_result.xlsx', 'rb')}
    # files = {'file': open('/home/ifly/cs/model/SenseVoiceSmall/example/zh.mp3', 'rb')}
    response = requests.post("http://localhost:8000/transcribe/", files=files)
    # 打印响应状态码
    print(f"Status Code: {response.status_code}")
    # 打印响应内容
    try:
        print("Response JSON:", response.json())
    except ValueError:
        print("Response is not in JSON format")

# 测试自我介绍评价接口
def test_self_intro():
    # data = {"introduction": "面试官，您好，是陈宋，来自安徽工业大学，学的是计算机科学技术专业。之前有一段实习在合肥数据空间研究院做的是大模型的评测相关的内容。"}
    # data = {"introduction": ""}
    data = {"introduction": "1212412412312我是程松"}
    headers = {'Content-Type': 'application/json'}  # 确保请求头设置为 application/json
    response = requests.post("http://localhost:8000/evaluate/self-introduction/", json=data,headers=headers)
    # 打印响应状态码
    print(f"Status Code: {response.status_code}")
    # 打印响应内容
    try:
        print("Response JSON:", response.json())
    except ValueError:
        print("Response is not in JSON format")


# 测试面试问答评价接口
def test_interview_answer():
    data = {"question": "请告诉我jvm的垃圾回收机制", "answer": "通过调用System.gc()方法来建议JVM进行垃圾回收。JVM会根据当前的内存使用情况和垃圾回收策略来决定是否进行垃圾回收。"}
    # data = {"question": "请告诉我jvm的垃圾回收机制", "answer": "垃圾"}
    # data = {"question": "", "answer": "通过调用System.gc()方法来建议JVM进行垃圾回收。JVM会根据当前的内存使用情况和垃圾回收策略来决定是否进行垃圾回收。"}
    # data = {"question": "请告诉我jvm的垃圾回收机制", "answer": ""}
    headers = {'Content-Type': 'application/json'}  # 确保请求头设置为 application/json
    response = requests.post("http://localhost:8000/evaluate/interview-answer/", json=data, headers=headers)
    # 打印响应状态码
    print(f"Status Code: {response.status_code}")
    # 打印响应内容
    try:
        print("Response JSON:", response.json())
    except ValueError:
        print("Response is not in JSON format")

# 运行测试
if __name__ == "__main__":
    # Uncomment to test each endpoint
    # test_transcribe()
    # test_transcribe_void()
    # test_self_intro()
    test_interview_answer()
    # from pydub import AudioSegment
    # def generate_silence(duration=0):
    #     silence = AudioSegment.silent(duration=duration)
    #     return silence
    # def save_as_mp3(audio, output_path):
    #     audio.export(output_path, format="mp3")
    # silence = generate_silence(5000)
    # save_as_mp3(silence, "silence.mp3")
