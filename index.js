document.addEventListener("DOMContentLoaded", () => {
    const splashText = document.getElementById("splash-text");
    const mainPage = document.getElementById("main-page");

    const texts = [
        "Namma Rakshane",
        "ನಮ್ಮ ರಕ್ಷಣೆ",
        "हमारी रक्षा",
        "మన రక్షణ"
    ];

    let index = 0;

    const displayText = () => {
        if (index < texts.length) {
            // Fade in the text
            splashText.style.opacity = 0; // Start invisible
            setTimeout(() => {
                splashText.innerHTML = `<h1>${texts[index]}</h1>`;
                splashText.style.opacity = 1; // Fade in
            }, 1000); // Delay before fade-in

            // Fade out the text
            setTimeout(() => {
                splashText.style.opacity = 0; // Fade out
                index++;
            }, 2000); // Stay visible for 3 seconds

            // Continue the animation
            setTimeout(displayText, 3000); // Total 4 seconds before next text
        } else {
            // Transition to the main page after all texts have been displayed
            setTimeout(() => {
                window.location.href = "main.html"; // Redirect to main.html
            }, 2000); // Wait for 2 seconds before going to the main page
        }
    };

    displayText();
});
