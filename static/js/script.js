document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('input_word').addEventListener('input', function () {
        fetchSuggestions();
    });

    function fetchSuggestions() {
        const inputWord = document.getElementById('input_word').value;

        fetch(`/suggest?input_text=${inputWord}`)
            .then(response => response.json())
            .then(data => {
                const suggestionsDiv = document.getElementById('suggestions');
                suggestionsDiv.innerHTML = '';

                if (data.suggestions.length > 0) {
                    data.suggestions.forEach(suggestion => {
                        const suggestionElement = document.createElement('div');
                        suggestionElement.innerText = suggestion[0];
                        suggestionElement.className = 'suggestion';
                        suggestionElement.onclick = function () {
                            document.getElementById('input_word').value += ' ' + suggestion[0];
                            suggestionsDiv.innerHTML = '';
                        };
                        suggestionsDiv.appendChild(suggestionElement);
                    });
                }
            })
            .catch(error => console.error('Error fetching suggestions:', error));
    }
});
