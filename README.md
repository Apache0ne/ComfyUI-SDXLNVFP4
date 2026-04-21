
working in ComfyUI with sdxl models:

Just need the two safetensors model1 and model2. clip_g and clip_l for the common name from sdxl

Needs my custom loader here to work: [https://github.com/Apache0ne/ComfyUI-SDXLNVFP4](https://github.com/Apache0ne/ComfyUI-SDXLNVFP4)

Workflow in images, also I am using taesd vae for both.

<table style="width: auto;">
  <tr>
    <!-- Image 1 -->
    <td align="center" valign="top" style="padding: 10px;">
      <a href="https://huggingface.co/ApacheOne/sdxl_text_encoders-NVFP4/blob/main/assets/ComfyUI_temp_bajmn_00003_.png" target="_blank">
        <img src="https://huggingface.co/ApacheOne/sdxl_text_encoders-NVFP4/blob/main/assets/ComfyUI_temp_bajmn_00003_.png" height="120" style="border: 1px solid #444; border-radius: 4px;">
      </a>
      <br>
      <p style="width: 150px; word-wrap: break-word; line-height: 1.2;">
        <small><code>With NVFP4 CLIP</code></small>
      </p>
    </td>
    <!-- Image 2 -->
    <td align="center" valign="top" style="padding: 10px;">
      <a href="https://huggingface.co/ApacheOne/sdxl_text_encoders-NVFP4/blob/main/assets/ComfyUI_temp_bajmn_00004_.png" target="_blank">
        <img src="https://huggingface.co/ApacheOne/sdxl_text_encoders-NVFP4/blob/main/assets/ComfyUI_temp_bajmn_00003_.png" height="120" style="border: 1px solid #444; border-radius: 4px;">
      </a>
      <br>
      <p style="width: 150px; word-wrap: break-word; line-height: 1.2;">
        <small><code>with normal CLIP</code></small>
      </p>
    </td>
  </tr>
</table>
