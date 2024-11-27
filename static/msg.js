$(document).ready(function () {
    $("#textForm").submit(function (e) {
        e.preventDefault();
        const userMessage = $("#text").val().trim();
        if (userMessage === "") return;

        $("#messageContainer").append(`<div class="user-message">${userMessage}</div>`);

        $.ajax({
            type: "POST",
            url: "/get_text_response",
            contentType: "application/json",
            data: JSON.stringify({ message: userMessage }),
            success: function (response) {
                $("#messageContainer").append(`<div class="bot-message">${response.response}</div>`);
            },
            error: function () {
                $("#messageContainer").append(`<div class="bot-message">Error processing your request.</div>`);
            }
        });
    });

    // Handle the submission of image upload
    $("#imageForm").submit(function (e) {
        e.preventDefault();
        const formData = new FormData();
        const file = $("#inputImage")[0].files[0];
    
        if (!file) {
            alert("Please select an image file.");
            return;
        }
    
        formData.append("image", file);
    
        $.ajax({
            type: "POST",
            url: "/upload_image",
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                // Display the detected style and the clothing advice
                $("#messageContainer").append(`<div class="bot-message">Detected Style: ${response.style}<br>Advice: ${response.advice}</div>`);
            },
            error: function () {
                $("#messageContainer").append(`<div class="bot-message">Error processing your request.</div>`);
            }
        });
    });

    // Handle video capture
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const captureBtn = document.getElementById("captureBtn");

    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
        video.srcObject = stream;
    });

    captureBtn.addEventListener("click", function () {
        const context = canvas.getContext("2d");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(function (blob) {
            const formData = new FormData();
            formData.append("image", blob, "captured_image.jpg");

            $.ajax({
                type: "POST",
                url: "/upload_image",
                data: formData,
                processData: false,
                contentType: false,
                success: function (response) {
                    // Display the detected style and advice from captured image
                    $("#messageContainer").append(
                        `<div class="bot-message">Style: ${response.style}<br>Advice: ${response.advice}</div>`
                    );
                },
                error: function () {
                    $("#messageContainer").append(
                        `<div class="bot-message">Error processing the captured image.</div>`
                    );
                },
            });
        });
    });
});
