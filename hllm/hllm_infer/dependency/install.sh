unset PIP_EXTRA_INDEX_URL
unset PIP_INDEX_URL
mkdir -p /root/.pip && \
    echo "[global]\n\
    index-url = https://mirrors.aliyun.com/pypi/simple\n\
    extra-index-url =\n\
    no-cache-dir = true\n\
    disable-pip-version-check = true\n\
    timeout = 300\n" > /root/.pip/pip.conf
python3 -m pip install transformers==4.50.1 --index-url https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com --timeout 100 
python3 -m pip install triton==3.4.0 --index-url https://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.aliyun.com --timeout 100 
