import os
from app import app

if __name__ == "__main__":
    # 获取环境变量中的端口，如果没有则使用5000
    port = int(os.environ.get("PORT", 5000))
    # 在生产环境中禁用调试模式
    debug = os.environ.get("FLASK_ENV", "production") != "production"
    # 监听所有接口，允许外部访问
    app.run(host="0.0.0.0", port=port, debug=debug)
