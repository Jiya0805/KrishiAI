from django.urls import path
from . import views

urlpatterns = [
    path('predict-crop/', views.predict_crop, name='predict_crop'),
    path('predict-plant-disease/', views.predict_plant_disease, name='predict_plant_disease'),
    path('predict-pest/', views.predict_pest, name='predict_pest'),
    path('health/', views.health_check, name='health_check'),
    path('calculate-profit/', views.calculate_profit, name='calculate_profit'),
    path('cultivation-roadmap/', views.generate_cultivation_roadmap, name='cultivation_roadmap'),
    path('chat/', views.chat_with_ai, name='chat_with_ai'),
    path('generate-pdf/', views.generate_prediction_pdf, name='generate_pdf'),
]
