async function predict() {
    let file = document.getElementById("imageInput").files[0];
    let model = document.getElementById("model").value;

    if (!file) {
        alert("Please upload an MRI image");
        return;
    }

    let formData = new FormData();
    formData.append("image", file);
    formData.append("model", model);

    document.getElementById("loader").style.display = "block";

    let res = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    let data = await res.json();

    document.getElementById("loader").style.display = "none";

    let container = document.getElementById("comparisonContainer");
    let heatmapContainer = document.getElementById("heatmapContainer");

    container.innerHTML = "";
    heatmapContainer.innerHTML = "";

    // ---------------- ORIGINAL ----------------
    container.innerHTML += `
    <div class="compare-box">
        <h4>Original MRI</h4>
        <img src="${URL.createObjectURL(file)}">
    </div>
    `;

    // ================= BOTH =================
    if (data.model === "Both") {

        if (data.unet_overlay) {
            container.innerHTML += `
            <div class="compare-box">
                <h4>U-Net</h4>
                <img src="${data.unet_overlay}">
            </div>`;
        }

        if (data.rcnn_overlay) {
            container.innerHTML += `
            <div class="compare-box">
                <h4>Mask R-CNN</h4>
                <img src="/${data.rcnn_overlay}">
            </div>`;
        }

        if (data.unet_heatmap) {
            heatmapContainer.innerHTML += `
            <div class="compare-box">
                <h4>U-Net Heatmap</h4>
                <img src="${data.unet_heatmap}">
            </div>`;
        }

        if (data.rcnn_heatmap) {
            heatmapContainer.innerHTML += `
            <div class="compare-box">
                <h4>Mask R-CNN Heatmap</h4>
                <img src="${data.rcnn_heatmap}">
            </div>`;
        }
    }

    // ================= SINGLE =================
    else if (data.model === "U-Net") {

        if (data.unet_overlay) {
            container.innerHTML += `
            <div class="compare-box">
                <h4>U-Net</h4>
                <img src="${data.unet_overlay}">
            </div>`;
        }

        if (data.unet_heatmap) {
            heatmapContainer.innerHTML += `
            <div class="compare-box">
                <h4>U-Net Heatmap</h4>
                <img src="${data.unet_heatmap}">
            </div>`;
        }
    } else if (data.model === "Mask R-CNN") {

        if (data.rcnn_overlay) {
            container.innerHTML += `
            <div class="compare-box">
                <h4>Mask R-CNN</h4>
                <img src="${data.rcnn_overlay}">
            </div>`;
        }

        if (data.rcnn_heatmap) {
            heatmapContainer.innerHTML += `
            <div class="compare-box">
                <h4>Mask R-CNN Heatmap</h4>
                <img src="${data.rcnn_heatmap}">
            </div>`;
        }
    }

    // ================= CONFIDENCE =================
    if (data.model === "Both") {
        document.getElementById("confidence").innerHTML = `
        <b>U-Net</b> → Confidence: ${data.confidence.unet?.toFixed(3) || 0} | Area: ${data.area.unet || 0} mm²<br>
        <b>Mask R-CNN</b> → Confidence: ${data.confidence.rcnn?.toFixed(3) || 0} | Area: ${data.area.rcnn || 0} mm²
        `;
    } else if (data.model === "U-Net") {
        document.getElementById("confidence").innerHTML = `
        <b>U-Net</b> → Confidence: ${data.confidence.unet?.toFixed(3) || 0} | Area: ${data.area.unet || 0} mm²
        `;
    } else if (data.model === "Mask R-CNN") {
        document.getElementById("confidence").innerHTML = `
        <b>Mask R-CNN</b> → Confidence: ${data.confidence.rcnn?.toFixed(3) || 0} | Area: ${data.area.rcnn || 0} mm²
        `;
    }
}

// PDF DOWNLOAD
function downloadPDF() {
    window.open("/download_report");
}