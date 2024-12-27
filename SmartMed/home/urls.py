from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index),
    path('articles/', views.articles),
    path('contactez-nous/', views.contact),
    path('services/', views.services),
    path('signup/',views.register, name='register'),
    path('login/',views.login_view, name='login'),
    path('segmentation/', views.segmentation, name='segmentation'),
    path('segment_image/', views.segment_image, name='segment_image'),
    path('image-processing/', views.upload_image, name='upload_image'),
    path('prediction/', views.prediction, name='prediction'),
    path('predict/', views.predict_disease, name='predict_disease'),
    path('iot/', views.iot, name='iot'),
    path('optimisation/', views.optimization_view, name='optimization_view'),
    path('contactez-nous/', views.contact_view, name='contact_view'),
    path('llm/', views.llm, name='llm'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


