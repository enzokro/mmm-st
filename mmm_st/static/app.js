var eventSource = new EventSource("/stream");
eventSource.onmessage = function(event) {
    document.getElementById('liveImage').src = 'data:image/jpeg;base64,' + event.data;
};
eventSource.onerror = function() {
    console.error("Error receiving stream");  // Handle errors
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

function showNonBlockingNotification(message, type) {
    // Simple example using a basic div or span to display a message
    var notification = document.createElement('div');
    notification.innerText = message;
    notification.className = 'notification ' + type;
    document.body.appendChild(notification);

    // Automatically remove the notification after a delay
    setTimeout(function() {
        document.body.removeChild(notification);
    }, 1500);  // Remove after 3 seconds
}

document.getElementById('refreshForm').addEventListener('click', function(e) {
    e.preventDefault();  // Prevent default form submission
    fetch('/refresh_prompt', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ signal: "refresh" }),
    })
    .then(response => response.json())  // Handle the response
    .then(data => {
        console.log('Success:', data);
        showNonBlockingNotification('Scene refreshed!', 'success');  // Non-blocking notification
    })
    .catch((error) => {
        console.error('Error:', error);
        showNonBlockingNotification('Could not refresh scene.', 'error');  // Non-blocking error notification
    });
});
