import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import run_app
import os
import re

app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'  # 세션을 위한 비밀 키 설정

@app.route('/')
def index():
    # return 'Hello World!'
    return render_template('index.html')

@app.route('/start-training', methods=['POST'])
def start_training():
    global fl_server
    # 파라미터 설정 및 서버 초기화
    params = {
        "rounds": request.form.get('rounds'),
        "clients": request.form.get('clients'),
        "dataset": request.form.get('dataset'),
        "data_distribution": request.form.get('data_distribution')
    }

    # file_logger, log_file_path = run_app.file_generate()
    fl_server, log_file_path = run_app.server_init(params=params)
    run_app.server_boot(fl_server)

    # 세션에 정보 저장
    session["params"] = params
    session['log_file_path'] = log_file_path
    session['current_round'] = 0
    session['total_round'] = int(params["rounds"])
    total_rounds = int(params["rounds"])

    return render_template('loading.html',total_rounds=total_rounds)

@app.route('/training-status')
def training_status():
    global fl_server
    current_round = session.get('current_round', 0)
    total_round = session.get('total_round', 0)

    if current_round < total_round:
        run_app.server_run(fl_server, current_round)
        session['current_round'] = current_round + 1  # 현재 라운드 증가
        session.modified = True  # 세션 변경 알림

    log_file_path = session.get('log_file_path', None)
    progress = get_training_progress(log_file_path)
    return jsonify({'progress': progress, 'current_round': current_round})

def get_training_progress(log_file_path):
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1]  # 마지막 라인 추출
                progress_info = last_line.split()  # 공백을 기준으로 분리
                return progress_info[0] if progress_info else "No progress"
            else:
                return "No progress yet"
    except FileNotFoundError:
        return "Log file not found"
    except Exception as e:
        return f"Error: {e}"





import pandas as pd

@app.route('/result')
def show_result():
    logs_folder_path = os.path.join(os.getcwd(), 'logs')
    sessions = get_previous_sessions()  # 이전 세션 목록 생성
    sessions_data = get_sessions_data(logs_folder_path)  # 각 세션의 데이터 생성

    return render_template('result.html', sessions=sessions, sessions_data=sessions_data)


def read_log_file(log_file_path):
    data = []
    with open(log_file_path, 'r') as file:
        for line in file:
            round, accuracy = line.strip().split('_')
            data.append([int(round), float(accuracy)])
    return data


def get_description_from_log_filename(filename):
    # 파일명에서 대괄호 안의 숫자만 추출하기 위한 정규 표현식
    rounds_match = re.search(r'\[(\d+)\]rounds', filename)
    clients_match = re.search(r'\[(\d+)\]clients', filename)
    dataset_match = re.search(r'\[([a-zA-Z]+)\]', filename)
    data_distribution_match = re.search(r'\[(\d+)\]IID', filename)

    rounds = rounds_match.group(1) if rounds_match else "unknown"
    clients = clients_match.group(1) if clients_match else "unknown"
    dataset = dataset_match.group(1) if dataset_match else "unknown"
    data_distribution = data_distribution_match.group(1) if data_distribution_match else "unknown"

    description = f"{rounds} rounds, {clients} clients, {dataset} dataset, {data_distribution} IID"
    return description

def get_previous_sessions():
    sessions = []
    log_directory = os.path.join(os.getcwd(), 'logs')
    for filename in sorted(os.listdir(log_directory)):
        if filename.endswith('.log'):
            description = get_description_from_log_filename(filename)
            sessions.append({'id': filename, 'description': description})
    return sessions


def get_sessions_data(logs_folder_path):
    sessions_data = {}
    for filename in os.listdir(logs_folder_path):
        if filename.endswith('.log'):
            # 로그 파일의 전체 경로
            file_path = os.path.join(logs_folder_path, filename)
            # 파일로부터 데이터를 읽어 DataFrame 생성
            data = read_log_file(file_path)
            df = pd.DataFrame(data, columns=['round', 'accuracy'])

            # 'round' 열로 그룹화하고, 각 그룹별 'accuracy'의 평균을 계산
            grouped_df = df.groupby('round')['accuracy'].mean().reset_index()

            # DataFrame을 JSON 형태로 변환
            sessions_data[filename] = grouped_df.to_json(orient='split')
    return sessions_data

if __name__ == '__main__':
    app.run(debug=True)
