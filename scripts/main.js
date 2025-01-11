// Check if a language is already selected
if (localStorage.getItem('language')) {
    // Redirect to gateway.html if a language is already selected
    window.location.href = "gateway.html";
}

// Function to save selected language and redirect
function selectLanguage(language) {
    localStorage.setItem('language', language);
    window.location.href = "gateway.html"; // Redirect to gateway.html
}
