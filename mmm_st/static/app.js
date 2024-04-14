var eventSource = new EventSource("/stream");
eventSource.onmessage = function(event) {
    document.getElementById('liveImage').src = 'data:image/jpeg;base64,' + event.data;
};

document.getElementById('promptForm').addEventListener('submit', function(e) {
    e.preventDefault();
    var promptValue = document.getElementById('promptInput').value;
    fetch('/set_prompt', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: promptValue }),
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
        alert('Prompt set successfully!');
    })
    .catch((error) => {
        console.error('Error:', error);
        alert('Failed to set prompt');
    });
});
