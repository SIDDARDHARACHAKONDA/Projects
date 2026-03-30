import json
import numpy as np
import joblib
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .models import DetectionHistory

def landing(request):
    return render(request, 'landing.html')


def register_view(request):

    if request.method == 'POST':

        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']

        # Check if username already exists
        if User.objects.filter(username=username).exists():
            return render(request, 'register.html', {
                'error': 'Username already exists'
            })

        # Check if email already exists
        if User.objects.filter(email=email).exists():
            return render(request, 'register.html', {
                'error': 'Email already registered'
            })

        # Create new user
        User.objects.create_user(
            username=username,
            email=email,
            password=password
        )

        return render(request, 'register.html', {
            'success': 'Registration successful! Please login.'
        })

    return render(request, 'register.html')


def login_view(request):

    if request.method == 'POST':

        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('/dashboard/')
        else:
            # Invalid login
            return render(request, 'login.html', {
                'error': 'Invalid username or password'
            })

    return render(request, 'login.html')


@login_required
def dashboard(request):
    return render(request, 'dashboard.html')


def logout_view(request):
    logout(request)
    return redirect('/')


import pandas as pd
from django.core.files.storage import FileSystemStorage

def upload_dataset(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('dataset'):
        file = request.FILES['dataset']

        fs = FileSystemStorage(location='media/')
        filename = fs.save(file.name, file)
        filepath = fs.path(filename)

        # read dataset using pandas
        df = pd.read_csv(filepath)

        # save path in session for next steps
        request.session['dataset_path'] = filepath

        # preview first 10 rows
        context['headers'] = list(df.columns)
        context['rows'] = df.head(10).values.tolist()

    return render(request, 'upload.html', context)


from sklearn.preprocessing import MinMaxScaler


def preprocess_dataset(request):
    context = {}

    if request.method == 'POST':
        filepath = request.session.get('dataset_path')

        if not filepath:
            context['error'] = "No dataset uploaded."
            return render(request, 'preprocess.html', context)

        df = pd.read_csv(filepath)

        # drop rows with missing values
        df = df.dropna()

        # separate label if exists
        if 'label' in df.columns:
            y = df['label']
            X = df.drop('label', axis=1)
        else:
            X = df

        # one-hot encode categorical columns
        X = pd.get_dummies(X)

        # scale
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        # save preprocessed data path for next steps
        processed_path = filepath.replace('.csv', '_processed.csv')
        X_scaled_df.to_csv(processed_path, index=False)
        request.session['processed_path'] = processed_path

        # preview first 10 rows
        context['headers'] = list(X_scaled_df.columns)
        context['rows'] = X_scaled_df.head(10).values.tolist()

    return render(request, 'preprocess.html', context)


import pandas as pd
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from django.contrib.auth.decorators import login_required
from django.shortcuts import render


def train_algorithms(request):
    context = {}

    if request.method == 'POST':
        filepath = request.session.get('processed_path')
        original_path = request.session.get('dataset_path')

        if not filepath or not original_path:
            context['error'] = "Upload and preprocess dataset first."
            return render(request, 'train.html', context)

        # load processed features
        X = pd.read_csv(filepath)

        # load original to get labels
        original_df = pd.read_csv(original_path)
        y = original_df['label'].loc[X.index]

        # split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        results = []
        compare_data = {}

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(n_estimators=80, max_depth=5, learning_rate=0.1, eval_metric="logloss"),
            "Neural Network": MLPClassifier(hidden_layer_sizes=(32,), max_iter=200)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred)
            rec = recall_score(y_test, pred)
            f1 = f1_score(y_test, pred)

            # confusion matrix
            cm = confusion_matrix(y_test, pred)

            # create heatmap image
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(name + " Confusion Matrix")

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)

            heatmap_base64 = base64.b64encode(buf.read()).decode('utf-8')

            results.append({
                "name": name,
                "acc": round(acc, 4),
                "prec": round(prec, 4),
                "rec": round(rec, 4),
                "f1": round(f1, 4),
                "heatmap": heatmap_base64
            })

            compare_data[name] = round(acc, 4)

        # save accuracies for comparison graph page
        request.session['compare_data'] = compare_data
        context['results'] = results

    return render(request, 'train.html', context)

import json

def compare_view(request):
    data = request.session.get('compare_data', {})
    return render(request, 'compare.html', {"data": json.dumps(data)})



import numpy as np
import joblib
import logging
import threading
from django.core.mail import send_mail
from django.conf import settings
from .models import DetectionHistory

logger = logging.getLogger(__name__)

@login_required
def detect_view(request):
    result = None

    if request.method == 'POST':
        try:
            values = [
                float(request.POST['dur']),
                float(request.POST['spkts']),
                float(request.POST['dpkts']),
                float(request.POST['sbytes']),
                float(request.POST['dbytes']),
                float(request.POST['rate']),
                float(request.POST['sttl']),
                float(request.POST['dttl']),
                float(request.POST['sload']),
                float(request.POST['dload']),
            ]

            X = np.array(values).reshape(1, -1)

            model = joblib.load("ml_training/ml_models/stacking_model.pkl")
            scaler = joblib.load("ml_training/ml_models/scaler.pkl")
            le = joblib.load("ml_training/ml_models/label_encoder.pkl")

            X_scaled = scaler.transform(X)

            pred_num = model.predict(X_scaled)[0]

            # Convert numeric prediction to label
            result = le.inverse_transform([pred_num])[0]

            # Save detection history
            DetectionHistory.objects.create(
                user=request.user,
                dur=values[0],
                spkts=values[1],
                dpkts=values[2],
                sbytes=values[3],
                dbytes=values[4],
                rate=values[5],
                sttl=values[6],
                dttl=values[7],
                sload=values[8],
                dload=values[9],
                prediction=result
            )


        except Exception as e:
            result = "Error: " + str(e)
        else:
            # Send email alert if attack detected
            if result != "Normal" and settings.ALERT_EMAIL_ENABLED:

                subject = "IoMT Security Alert"

                message = f"""
Hello Admin,

An attack has been detected in the IoMT Network.

User: {request.user.username}
Attack Type: {result}

Network Features:
dur={values[0]}
spkts={values[1]}
dpkts={values[2]}
sbytes={values[3]}
dbytes={values[4]}
rate={values[5]}

Please investigate immediately.

IoMT Security System
"""

                def _send_alert_email():
                    try:
                        send_mail(
                            subject,
                            message,
                            settings.EMAIL_HOST_USER,
                            ['admin@example.com'],  # admin email
                            fail_silently=False,
                        )
                    except Exception as e:
                        logger.warning("Email alert failed: %s", e, exc_info=True)

                # Send email in background to avoid request timeouts in production.
                threading.Thread(target=_send_alert_email, daemon=True).start()

    return render(request, 'detect.html', {"result": result})

@login_required
def detect_batch_view(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('csv_file'):
        file = request.FILES['csv_file']

        df = pd.read_csv(file)

        # ⚠️ MUST match training features exactly
        feature_cols = [
            'dur','spkts','dpkts','sbytes','dbytes',
            'rate','sttl','dttl','sload','dload'
        ]

        X = df[feature_cols]

        model = joblib.load("ml_training/ml_models/stacking_model.pkl")
        scaler = joblib.load("ml_training/ml_models/scaler.pkl")
        le = joblib.load("ml_training/ml_models/label_encoder.pkl")

        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        labels = le.inverse_transform(preds)

        results = []
        for i in range(len(df)):
            results.append({
                "data": X.iloc[i].values.tolist(),
                "pred": labels[i]
            })

        context["headers"] = feature_cols
        context["results"] = results

    return render(request, "detect_batch.html", context)

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

def admin_login(request):

    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        if username == "admin" and password == "admin":
            request.session['admin'] = True
            return redirect("/admin-dashboard/")

        else:
            return render(request, "admin_login.html", {"error": "Invalid Admin Credentials"})

    return render(request, "admin_login.html")


def admin_dashboard(request):

    if not request.session.get('admin'):
        return redirect("/admin-login/")

    return render(request, "admin_dashboard.html")


def admin_logout(request):
    request.session.flush()
    return redirect("/admin-login/")

def admin_users(request):

    if not request.session.get('admin'):
        return redirect('/admin-login/')

    users = User.objects.all()

    return render(request, "admin_users.html", {"users": users})

def delete_user(request, user_id):

    if not request.session.get('admin'):
        return redirect('/admin-login/')

    user = User.objects.get(id=user_id)
    user.delete()

    return redirect('/admin-users/')

from .models import DetectionHistory


def admin_history(request):

    if not request.session.get('admin'):
        return redirect('/admin-login/')

    history = DetectionHistory.objects.all().order_by('-created_at')

    return render(request, "admin_history.html", {"history": history})


from django.db.models import Count
from .models import DetectionHistory
import json


def prediction_analysis(request):

    data = DetectionHistory.objects.values('prediction').annotate(
        total=Count('prediction')
    )

    labels = []
    values = []

    for i in data:
        labels.append(i['prediction'])
        values.append(i['total'])

    context = {
        "labels": json.dumps(labels),
        "values": json.dumps(values)
    }

    return render(request, "prediction_analysis.html", context)
