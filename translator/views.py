from django.http.response import StreamingHttpResponse
from django.shortcuts import render

from translator.services import VideoCamera


def index(request):
    return render(request, 'translator/index.html')


def video_feed(request):
    video_camera = VideoCamera()

    return StreamingHttpResponse(video_camera.gen(),
                                content_type='multipart/x-mixed-replace; boundary=frame')
