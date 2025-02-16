document.getElementById('summarizeForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const youtubeUrl = document.getElementById('youtube_url').value;
    const summaryResult = document.getElementById('summaryResult');
    const loading = document.getElementById('loading');

    summaryResult.innerHTML = '';
    loading.style.display = 'block';

    fetch('/summarize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ youtube_url: youtubeUrl }),
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';
        if (data.error) {
            summaryResult.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
        } else {
            summaryResult.innerHTML = `<p><strong>Summary:</strong> ${data.summary}</p>`;
        }
    })
    .catch(error => {
        loading.style.display = 'none';
        summaryResult.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    });
});