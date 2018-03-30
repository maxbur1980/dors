from django.urls import path

from translator import views


app_name = 'translator'

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
]
