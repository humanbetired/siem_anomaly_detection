import json
import requests
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from datetime import datetime


def index(request):
    return render(request, 'dashboard/index.html')


@csrf_exempt
def ingest_flow(request):
    """Terima flow dari client, forward ke FastAPI, broadcast alert."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        payload = json.loads(request.body)

        # Forward ke FastAPI
        response = requests.post(
            f"{settings.FASTAPI_URL}/ingest",
            json=payload,
            timeout=10
        )
        result = response.json()

        # Broadcast via WebSocket kalau anomali
        if result.get('is_anomaly'):
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                "alerts",
                {
                    "type": "alert_message",
                    "data": {
                        "type": "alert",
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "severity": result["severity"],
                        "anomaly_score": result["anomaly_score"],
                        "isolation_forest": result["isolation_forest_result"],
                        "autoencoder": result["autoencoder_result"],
                        "reconstruction_error": result["reconstruction_error"],
                        "message": result["message"]
                    }
                }
            )

        return JsonResponse(result)

    except requests.exceptions.ConnectionError:
        return JsonResponse(
            {'error': 'ML service unavailable'}, status=503
        )
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def get_stats(request):
    """Proxy stats dari FastAPI."""
    try:
        response = requests.get(
            f"{settings.FASTAPI_URL}/stats", timeout=5
        )
        return JsonResponse(response.json())
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)