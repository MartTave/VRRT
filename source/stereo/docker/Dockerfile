FROM pytorch/pytorch:latest
RUN git clone https://github.com/NVlabs/FoundationStereo && cd FoundationStereo && pip install -r requirements.txt
CMD ["python", "scripts/run_demo.py", "--left", "/data/left.png", "--right", "/data/right.png"]
