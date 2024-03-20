
    • pip install paddlepaddle-gpu==2.6.0.post120 -f https://www.paddlepaddle.org.cn/whl/linux/cudnnin/stable.html
    • sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
    • sudo ln -s /home/geyingke/anaconda3/envs/f4/lib/python3.9/site-packages/nvidia/cublas/lib/libcublas.so.12 /usr/lib/x86_64-linux-gnu/libcublas.so
    • export LD_LIBRARY_PATH=/home/geyingke/anaconda3/envs/f4/lib/python3.9/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH

问题出在安装paddle时，不会自动安装cuda， cudnn等库，虽然安装文档中说有会安装。
在安装pytorch包时，会自动安装，所以没有这个问题

参考：
https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/environment.md
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html




python ./script/run/t_easyocr.py 
export PYTHONPATH=./

[2024/01/19 10:29:39] ppocr DEBUG: dt_boxes num : 4, elapsed : 1.2661986351013184
[2024/01/19 10:29:39] ppocr DEBUG: cls num  : 4, elapsed : 0.00880575180053711
[2024/01/19 10:29:39] ppocr DEBUG: rec_res num  : 4, elapsed : 0.030220746994018555
[[[372.0, 68.0], [586.0, 68.0], [586.0, 129.0], [372.0, 129.0]], ('智能咨询', 0.9976475238800049)]
[[[1823.0, 72.0], [2010.0, 72.0], [2010.0, 118.0], [1823.0, 118.0]], ('问法规范', 0.9946289658546448)]
[[[129.0, 299.0], [1846.0, 299.0], [1846.0, 336.0], [129.0, 336.0]], ('HI！我是京京，很高兴为您服务～您可以向我咨询关于北京市政务服务事项的问题，也可以选择写信或拨打12345热线进行', 0.9897176027297974)]
[[[132.0, 355.0], [252.0, 355.0], [252.0, 395.0], [132.0, 395.0]], ('咨询哦~', 0.9372810125350952)]


(multi_model) geyingke@CP-001:~/anaconda3/envs/multi_model/lib/python3.9/site-packages$ pip install PyMuPDF==1.20.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 

ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/geyingke/anaconda3/envs/multi_model/bin/../lib/libstdc++.so.6

验证paddle
paddle.utils.require_version(  paddle.utils.run_check()


macOS 不支持

上述命令默认安装avx、mkl的包，判断㇏你的机器是否支持avx，可以输入以下命令，如果输出中包含avx，则表示机器支持avx。飞桨不再支持noavx指令集的安装包。
sysctl machdep.cpu.features | grep -i avx

或
sysctl machdep.cpu.leaf7_features | grep -i avx
