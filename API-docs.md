# Image Prediction API

This API provides image classification and bounding box prediction using either a Vision Transformer (ViT) or a Random Forest model.

## API Endpoints

### POST /predict

Performs inference on an input image to predict its class and bounding box.

## Request Body Schema

| Field   | Type     | Description                                                          | Required |
| :------ | :------- | :------------------------------------------------------------------- | :------- |
| `image` | `string` | Base64 encoded string of the input image. (JPEG or PNG recommended). | Yes      |
| `model` | `enum`   | Model to use for prediction. Accepted values: ViT, Forest.           | Yes      |

Example Request Body:

```json
{
	"image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
	"model": "ViT"
}
```

## Response Body Schema

| Field         | Type           | Description                                                                                                 |
| :------------ | :------------- | :---------------------------------------------------------------------------------------------------------- |
| `species`     | `string`       | The predicted Bird species.                                                                                 |
| `ounding_box` | `array<float>` | Predicted bounding box coordinates for the Bird [x\_min, y\_min, x\_max, y\_max] scaled to the [0-1] range. |

Example Successful Response (200 OK):

```json
{
	"species": "Herring Gull",
	"bounding_box": [0.1, 0.2, 0.8, 0.9]
}
```

## Possible Responses

| Status Code | Description                                                                       | Content Type     | Example                                                                                                                                                                |
| :---------- | :-------------------------------------------------------------------------------- | :--------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `200`       | OK: Prediction successful.                                                        | application/json | See example above.                                                                                                                                                     |
| `400`       | Bad Request: Invalid base64 string, malformed image data, or preprocessing error. | application/json | {"detail": "Invalid base64 string. Ensure the image is correctly encoded."} or {"detail": "Malformed image data or unsupported image format. Error: [error message]"}` |
| `422`       | Unprocessable Entity: Validation error due to incorrect input schema.             | application/json | {"detail": "Field "image" is required."} or {"detail": "Value is not a valid enumeration member; permitted: "ViT", "Forest""}                                          |

## Example API call

```python
import base64
import requests
import json

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

payload = {"image": image_file_path, "model": "ViT"}
headers = {"Content-Type": "application/json"}

response = requests.post("http://localhost:8000/predict", headers=headers, data=json.dumps(payload))

print(f"Status Code: {response.status_code}")
print(f"Response JSON: {response.json()}")
```
