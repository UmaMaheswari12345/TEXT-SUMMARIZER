function summarizeText() {
    let text = document.getElementById("text-input").value;
    let numSentences = document.getElementById("num-sentences").value;

    if (!text.trim()) {
        alert("Please enter some text.");
        return;
    }

    fetch("/summarize", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: text, num_sentences: numSentences })
    })
    .then(response => response.json())
    .then(data => {
        if (data.summary) {
            document.getElementById("summary-output").innerText = data.summary;
        } else {
            document.getElementById("summary-output").innerText = "No summary returned.";
        }
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("summary-output").innerText = "Error: Could not get summary.";
    });
}
