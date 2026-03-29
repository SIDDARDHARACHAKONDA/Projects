from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing),
    path('register/', views.register_view),
    path('login/', views.login_view),
    path('dashboard/', views.dashboard),
    path('logout/', views.logout_view),

    # temporary placeholders for next pages
    path('upload/', views.upload_dataset,),
    path('preprocess/', views.preprocess_dataset),
    path('train/', views.train_algorithms),
    path('compare/', views.compare_view),
    path('detect/', views.detect_view),
    path('detect-batch/', views.detect_batch_view),
    path("admin-login/", views.admin_login),
path("admin-dashboard/", views.admin_dashboard),
path("admin-logout/", views.admin_logout),
path('admin-users/', views.admin_users),
path('delete-user/<int:user_id>/', views.delete_user),

path('admin-history/', views.admin_history),
path('prediction-analysis/', views.prediction_analysis),
]
