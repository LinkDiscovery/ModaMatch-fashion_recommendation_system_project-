from flask import Flask

def create_app():
    # Flask 애플리케이션 인스턴스를 생성합니다.
    app = Flask(__name__)

    # 업로드된 파일을 저장할 디렉토리를 설정합니다.
    # 'uploads' 폴더를 사용하여 업로드된 파일을 저장할 것입니다.
    UPLOAD_FOLDER = './app/uploads'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # main.py 모듈에서 main 블루프린트를 가져옵니다.
    # 블루프린트는 Flask 애플리케이션의 구성 요소를 모듈화하는 데 사용됩니다.
    from .main import main as main_blueprint

    # 애플리케이션에 블루프린트를 등록합니다.
    # 이렇게 하면 main 블루프린트에 정의된 모든 라우트가 애플리케이션에 추가됩니다.
    app.register_blueprint(main_blueprint)

    # 설정 및 등록이 완료된 Flask 애플리케이션 인스턴스를 반환합니다.
    return app