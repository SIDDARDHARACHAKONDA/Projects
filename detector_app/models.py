from django.db import models
from django.contrib.auth.models import User


class DetectionHistory(models.Model):

    user = models.ForeignKey(User, on_delete=models.CASCADE)

    dur = models.FloatField()
    spkts = models.FloatField()
    dpkts = models.FloatField()
    sbytes = models.FloatField()
    dbytes = models.FloatField()
    rate = models.FloatField()
    sttl = models.FloatField()
    dttl = models.FloatField()
    sload = models.FloatField()
    dload = models.FloatField()

    prediction = models.CharField(max_length=100)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.prediction