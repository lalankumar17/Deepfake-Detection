from django import forms

class VideoUploadForm(forms.Form):

    upload_video_file = forms.FileField(label="Select File", required=True,widget=forms.FileInput(attrs={"accept": "video/*,image/*", "class": "form-control-file"}))
    sequence_length = forms.IntegerField(label="Sequence Length", required=True)
    use_gemini_review = forms.BooleanField(label="Gemini 2nd Verification", required=False)
