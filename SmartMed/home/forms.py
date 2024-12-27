from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class OptimizationForm(forms.Form):
    mA_min = forms.FloatField(label="mA min", min_value=0)
    time_min = forms.FloatField(label="Time min (s)", min_value=0)
    rpm_min = forms.FloatField(label="RPM min", min_value=0)
    mA_max = forms.FloatField(label="mA max", min_value=0)
    time_max = forms.FloatField(label="Time max (s)", min_value=0)
    rpm_max = forms.FloatField(label="RPM max", min_value=0)

class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=100, required=True)
    last_name = forms.CharField(max_length=100, required=True)

    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email', 'password1','password2')
    
    def save(self, commit=True):
        user = super(RegisterForm, self).save(commit=False)
        user.email = self.cleaned_data['email']
        user.username = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']

        if commit:
            user.save()
        
        return user