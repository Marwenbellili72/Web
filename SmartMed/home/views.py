from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, 'home/index.html')

def articles(request):
    return render(request, 'home/articles.html')

def contact(request):
    return render(request, 'home/contactez-nous.html') 

def services(request):
    return render(request, 'home/services.html') 

def llm(request):
    return render(request, 'home/llm.html') 

#################signup +++ Login######################
from django.shortcuts import render, redirect
from .forms import RegisterForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required

def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return render(request, 'home/login.html', {'success': "Registration successful. Please login."})
        else:
            error_message = form.errors.as_text()
            return render(request, 'home/signup.html', {'error': error_message})

    return render(request, 'home/signup.html')


def login_view(request):
    if request.method=="POST":
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            return redirect("/articles")
        else:
            return render(request, 'home/login.html', {'error': "Invalid credentials. Please try again."})

    return render(request, 'home/login.html')

#################form######################
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.core.mail import send_mail
from django.conf import settings

@require_POST
def contact_view(request):
    try:
        # Récupérer les données du formulaire
        name = request.POST.get('name', '').strip()
        email = request.POST.get('email', '').strip()  # Correction ici
        message = request.POST.get('message', '').strip()  # Correction ici

        # Validation des données
        if not all([name, email, message]):
            return JsonResponse({
                'success': False, 
                'error': 'Tous les champs sont obligatoires.'
            }, status=400)

        # Envoi de l'email (assurez-vous d'avoir configuré les paramètres d'email dans settings.py)
        send_mail(
            f'Nouveau message de {name}',
            f'De : {name} ({email})\n\nMessage : {message}',
            settings.DEFAULT_FROM_EMAIL,
            [settings.CONTACT_EMAIL],
            fail_silently=False,
        )

        return JsonResponse({
            'success': True, 
            'message': 'Votre message a été envoyé avec succès !'
        })
    
    except Exception as e:
        # Log l'erreur côté serveur si nécessaire
        return JsonResponse({
            'success': False, 
            'Error': 'Une erreur est survenue lors de l\'envoi du message.'
        }, status=500)

#################Segmentation######################
import os
import shutil
import nibabel as nib
import pyvista as pv
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from totalsegmentator.python_api import totalsegmentator

@login_required(login_url='/login/')
def segmentation(request):
    """Render the index page for segmentation"""
    return render(request, 'home/segmentation.html')

def cleanup_folders(input_folder, output_folder):
    """Clean up input and output folders."""
    for folder in [input_folder, output_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

@csrf_exempt
def segment_image(request):
    try:
        print("Requête POST reçue")
        if request.method == 'POST':
            print("Traitement du fichier...")
            # Vérifiez si le fichier est bien présent
            if 'nifti_file' not in request.FILES:
                return JsonResponse({'status': 'error', 'message': 'No file uploaded'}, status=400)
            uploaded_file = request.FILES['nifti_file']
            print(f"Fichier reçu : {uploaded_file.name}")

            # Save uploaded file
            input_folder = os.path.join('mediafiles', 'input')
            file_path = os.path.join(input_folder, uploaded_file.name)

            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            print(f"Fichier enregistré à {file_path}")

            # Maintenant, effectuez la segmentation
            output_folder = os.path.join('mediafiles', 'output')
            shutil.rmtree(output_folder, ignore_errors=True)  # Clean previous output
            totalsegmentator(file_path, output_folder, fast=True, verbose=True, roi_subset=["heart"])

            segmented_file = os.path.join(output_folder, 'heart.nii.gz')
            print(f"Segmentation terminée. Fichier : {segmented_file}")

            # Charger l'image segmentée
            img = nib.load(segmented_file)
            data = img.get_fdata()
            print(f"Image segmentée chargée, shape : {data.shape}")

            # 3D Visualization code (skip or add more debugging here)

            # Retourner la réponse JSON
            return JsonResponse({
                'status': 'success',
                'message': 'Segmentation completed',
                'segmented_file_path': '/media/output/heart.nii.gz'  # Retourner le chemin pour le téléchargement
            })

    except Exception as e:
        print(f"Erreur dans la segmentation : {str(e)}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

#################Amélioration des Images Médicales######################
from django.shortcuts import render
from django.http import HttpResponse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from io import BytesIO
from django.core.files.storage import FileSystemStorage
import base64

# Function to process image (load and apply transformations)
def process_image(image_path: str):
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Failed to load image '{image_path}'.")
        return None

    # Apply median filter and histogram equalization
    median_image = cv2.medianBlur(image, 5)  # Median filter with kernel size 5
    equalized_image = cv2.equalizeHist(image)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image)

    return image, median_image, equalized_image, clahe_image

# Function to plot histograms and return as BytesIO object
def plot_histograms(original, clahe):
    # Calculate histograms for original and CLAHE images
    hist_original = cv2.calcHist([original], [0], None, [256], [0, 256])
    hist_clahe = cv2.calcHist([clahe], [0], None, [256], [0, 256])

    plt.figure(figsize=(12, 4))

    # Plot original image histogram
    plt.subplot(1, 2, 1)
    plt.plot(hist_original, color='blue')
    plt.title('Original Image Histogram')

    # Plot CLAHE image histogram
    plt.subplot(1, 2, 2)
    plt.plot(hist_clahe, color='blue')
    plt.title('CLAHE Image Histogram')

    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Function to convert image (ndarray) to base64
def convert_image_to_base64(image: np.ndarray):
    # Convert numpy array to PNG format in memory
    _, buffer = cv2.imencode('.png', image)
    # Convert the binary buffer to base64
    return base64.b64encode(buffer).decode('utf-8')

# View function to handle image upload and processing
@login_required
def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        uploaded_file_url = fs.url(filename)

        # Process the uploaded image
        image_path = os.path.join('mediafiles', filename)
        result = process_image(image_path)

        if result:
            original_image, median_image, equalized_image, clahe_image = result

            # Convert images (ndarray) to base64
            original_image_base64 = convert_image_to_base64(original_image)
            median_image_base64 = convert_image_to_base64(median_image)
            equalized_image_base64 = convert_image_to_base64(equalized_image)
            clahe_image_base64 = convert_image_to_base64(clahe_image)

            # Generate histograms plot and convert to base64
            hist_buf = plot_histograms(original_image, clahe_image)
            histogram_base64 = base64.b64encode(hist_buf.getvalue()).decode('utf-8')

            # Render the result page with images and histogram in base64
            return render(request, 'home/result.html', {
                'original_image': original_image_base64,
                'median_image': median_image_base64,
                'equalized_image': equalized_image_base64,
                'clahe_image' : clahe_image_base64,
                'histogram': histogram_base64,
                'uploaded_file_url': uploaded_file_url
            })

    return render(request, 'home/upload.html')
#################Prédiction######################
import os
import joblib
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

# Charger le modèle ML
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'home/models/heart_disease.pkl')
models = joblib.load(model_path)
logistic_model = models['Logistic Regression']

@login_required
def prediction(request):
    return render(request, 'home/predict.html')

@csrf_exempt
def predict_disease(request):
    if request.method == "POST":
        try:
            # Récupérer les données JSON
            body = json.loads(request.body.decode('utf-8'))
            features = body.get('features', [])

            # Affichage pour vérifier les données reçues
            print("Received features:", features)

            if features:
                # Assurez-vous que vous n'envoyez que les 13 premières caractéristiques
                # Retirer la première valeur incorrecte (par exemple, une chaîne non valide)
                features = features[0][1:]  # Retirer le premier élément (qui est une chaîne non valide)

                # Convertir chaque élément de la liste en float ou int si possible
                features = [float(f) if f.replace('.', '', 1).isdigit() else 0 for f in features]

                # Affichage pour vérifier les features après conversion
                print("Features after conversion:", features)

                # Convertir en tableau 2D pour la prédiction
                features = np.array(features).reshape(1, -1)

                # Affichage pour vérifier le format final des features
                print("Features reshaped:", features)

                # Effectuer la prédiction
                prediction = logistic_model.predict(features)

                # Affichage pour vérifier la prédiction
                print("Prediction:", prediction)

                # Convertir la prédiction en int avant de la retourner
                prediction = int(prediction[0])

                # Retourner la prédiction au format JSON
                return JsonResponse({'prediction': prediction})
            else:
                return JsonResponse({'error': 'Invalid input data'}, status=400)
        
        except Exception as e:
            # Si une erreur se produit, l'afficher dans les logs
            print(f"Error: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid HTTP method'}, status=405)

#################IOT######################
@login_required
def iot(request):
    return render(request, 'home/iot.html', {'name': request.user.first_name}) 

#################optimization######################
from django.shortcuts import render
from .forms import OptimizationForm
from .optimizer import CTScannerOptimization
@login_required
def optimization_view(request):
    if request.method == "POST":
        form = OptimizationForm(request.POST)
        if form.is_valid():
            # Récupérer les bornes saisies par l'utilisateur
            mA_min = form.cleaned_data['mA_min']
            time_min = form.cleaned_data['time_min']
            rpm_min = form.cleaned_data['rpm_min']
            mA_max = form.cleaned_data['mA_max']
            time_max = form.cleaned_data['time_max']
            rpm_max = form.cleaned_data['rpm_max']

            # Créer une instance de l'optimisation et passer les bornes
            optimizer = CTScannerOptimization(
                lower_bounds=[mA_min, time_min, rpm_min],
                upper_bounds=[mA_max, time_max, rpm_max]
            )
            # Exécuter l'optimisation
            result = optimizer.run_optimization()

            # Afficher les résultats dans le template
            return render(request, 'home/results.html', {'result': result, 'form': form})
    else:
        form = OptimizationForm()

    return render(request, 'home/optimization_form.html', {'form': form})
