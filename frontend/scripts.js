function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() === "") return;

    appendMessage('user', userInput);

    fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: userInput }),
    })
    .then(response => response.json())
    .then(data => {
        appendMessage('assistant', data.answer);
    })
    .catch(error => {
        console.error('Error:', error);
        appendMessage('assistant', '抱歉，处理您的请求时出现错误。');
    });

    document.getElementById('user-input').value = '';
}

function appendMessage(sender, message) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.className = sender;
    messageElement.innerText = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}
