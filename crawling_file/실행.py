import subprocess

# 스크립트 파일 목록
scripts = [

    "크롤링하의(안나).py",
    "크롤링아우터(안나).py"
    "크롤링원피스(안나).py"
]

# 각 스크립트를 차례대로 실행
for script in scripts:
    try:
        # 각 스크립트 파일을 현재 디렉토리에서 실행
        result = subprocess.run(["python", script], check=True)
        print(f"{script} 실행 완료")
    except subprocess.CalledProcessError as e:
        print(f"{script} 실행 중 에러 발생: {e}")
        # 에러 발생 시에도 다음 스크립트 계속 실행
        continue