const predictBtn = document.getElementById("predict-btn");
const resultDiv = document.getElementById("result");

predictBtn.addEventListener("click", async function(event) {
  event.preventDefault();
  const sepalLength = document.getElementById("sepal_length").value;
  const sepalWidth = document.getElementById("sepal_width").value;
  const petalLength = document.getElementById("petal_length").value;
  const petalWidth = document.getElementById("petal_width").value;
  
  const response = await fetch("/predict", {
  method: "POST",
  headers: {
  "Content-Type": "application/json"
  },
  body: JSON.stringify({
  sepal_length: sepalLength,
  sepal_width: sepalWidth,
  petal_length: petalLength,
  petal_width: petalWidth
  })
  });
  
  const prediction = await response.json();
  resultDiv.innerHTML = Prediction: ${prediction.prediction};
  });