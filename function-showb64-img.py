"""
title: Base64 Image Carousel Action
author: OpenWebUI Assistant
version: 1.0.0
required_open_webui_version: 0.3.9
"""

from pydantic import BaseModel, Field
from typing import Optional
import re
import base64
import asyncio


class Action:
    class Valves(BaseModel):
        max_images: int = Field(
            default=10, description="Maximum number of images to display in carousel"
        )
        enable_debug: bool = Field(
            default=False, description="Enable debug output for troubleshooting"
        )
        auto_activate: bool = Field(
            default=True,
            description="Automatically activate when base64 images are detected",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:

        if not __event_emitter__:
            return None

        # Get the last message content
        try:
            last_message = body["messages"][-1]["content"]
        except (IndexError, KeyError):
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "No messages found", "done": True},
                }
            )
            return None

        # Extract base64 images using regex patterns
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Scanning for base64 images...", "done": False},
            }
        )

        # Multiple regex patterns to catch different base64 image formats
        patterns = [
            r"data:image/([^;]+);base64,([A-Za-z0-9+/]+={0,2})",  # Standard data URI
            r"!\[.*?\]\(data:image/([^;]+);base64,([A-Za-z0-9+/]+={0,2})\)",  # Markdown format
            r'<img[^>]*src=["\']data:image/([^;]+);base64,([A-Za-z0-9+/]+={0,2})["\'][^>]*>',  # HTML img tag
        ]

        base64_images = []

        for pattern in patterns:
            matches = re.finditer(pattern, last_message)
            for match in matches:
                if len(match.groups()) >= 2:
                    image_type = match.groups()[0]
                    image_data = match.groups()[1]

                    # Validate base64 data
                    try:
                        base64.b64decode(image_data, validate=True)
                        full_data_uri = f"data:image/{image_type};base64,{image_data}"
                        base64_images.append(full_data_uri)

                        if self.valves.enable_debug:
                            print(f"Found valid base64 image: {image_type}")

                    except Exception as e:
                        if self.valves.enable_debug:
                            print(f"Invalid base64 data: {str(e)}")
                        continue

        # Remove duplicates while preserving order
        base64_images = list(dict.fromkeys(base64_images))

        # Limit number of images
        if len(base64_images) > self.valves.max_images:
            base64_images = base64_images[: self.valves.max_images]

        if not base64_images:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "No base64 images found in message",
                        "done": True,
                    },
                }
            )
            return None

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": f"Found {len(base64_images)} images, generating carousel...",
                    "done": False,
                },
            }
        )

        # Generate the carousel HTML
        carousel_html = self.generate_carousel_html(base64_images)

        # Create artifact
        await __event_emitter__({"type": "message", "data": {"content": carousel_html}})

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Image carousel generated successfully!",
                    "done": True,
                },
            }
        )

        return None

    def generate_carousel_html(self, images):
        """Generate responsive image carousel HTML with touch/swipe support"""

        if len(images) == 1:
            # Single image display
            return f"""
            
            <div style="max-width: 800px; margin: 20px auto; text-align: center; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                <div style="background: #f8f9fa; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <img src="{images[0]}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);" alt="Base64 Image">
                    <p style="margin-top: 15px; color: #666; font-size: 14px;">Image extracted from message</p>
                </div>
            </div>
            """

        # Multi-image carousel
        image_elements = ""
        for i, img in enumerate(images):
            display_style = "block" if i == 0 else "none"
            image_elements += f"""
                   
                <div class="carousel-slide" style="display: {display_style};">
                    <img src="{img}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);" alt="Image {i+1}">
                </div>
            """

        # Navigation dots
        dots = ""
        for i in range(len(images)):
            active_class = "active" if i == 0 else ""
            dots += f'<span class="dot {active_class}" onclick="currentSlide({i+1})"></span>'

        return f"""
```html
        <div style="max-width: 800px; margin: 20px auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <div class="carousel-container" style="position: relative; background: #f8f9fa; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                
                <!-- Carousel Header -->
                <div style="text-align: center; margin-bottom: 15px;">
                    <h3 style="margin: 0; color: #333; font-size: 18px;">Image Carousel</h3>
                    <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{len(images)} images found</p>
                </div>
                
                <!-- Image Container -->
                <div class="carousel-images" style="position: relative; text-align: center; margin-bottom: 20px; touch-action: pan-y;">
                    {image_elements}
                    
                    <!-- Navigation Arrows -->
                    <button class="prev-btn" onclick="changeSlide(-1)" style="position: absolute; top: 50%; left: 10px; transform: translateY(-50%); background: rgba(0,0,0,0.5); color: white; border: none; border-radius: 50%; width: 40px; height: 40px; cursor: pointer; font-size: 18px; display: flex; align-items: center; justify-content: center; transition: background 0.3s;">‹</button>
                    <button class="next-btn" onclick="changeSlide(1)" style="position: absolute; top: 50%; right: 10px; transform: translateY(-50%); background: rgba(0,0,0,0.5); color: white; border: none; border-radius: 50%; width: 40px; height: 40px; cursor: pointer; font-size: 18px; display: flex; align-items: center; justify-content: center; transition: background 0.3s;">›</button>
                </div>
                
                <!-- Navigation Dots -->
                <div class="carousel-dots" style="text-align: center;">
                    {dots}
                </div>
                
                <!-- Image Counter -->
                <div class="image-counter" style="text-align: center; margin-top: 10px; color: #666; font-size: 14px;">
                    <span id="current-image">1</span> / {len(images)}
                </div>
            </div>
        </div>

        <style>
            .dot {{
                height: 12px;
                width: 12px;
                margin: 0 4px;
                background-color: #bbb;
                border-radius: 50%;
                display: inline-block;
                cursor: pointer;
                transition: all 0.3s;
            }}
            
            .dot.active, .dot:hover {{
                background-color: #007bff;
                transform: scale(1.2);
            }}
            
            .prev-btn:hover, .next-btn:hover {{
                background: rgba(0,0,0,0.7) !important;
                transform: translateY(-50%) scale(1.1);
            }}
            
            .carousel-slide {{
                transition: opacity 0.5s ease-in-out;
            }}
            
            @media (max-width: 600px) {{
                .prev-btn, .next-btn {{
                    width: 35px !important;
                    height: 35px !important;
                    font-size: 16px !important;
                }}
            }}
        </style>

        <script>
            let currentSlideIndex = 0;
            const totalSlides = {len(images)};
            
            // Touch/swipe functionality
            let startX = 0;
            let endX = 0;
            let isDragging = false;
            
            const carouselContainer = document.querySelector('.carousel-images');
            
            // Touch events
            carouselContainer.addEventListener('touchstart', handleTouchStart, {{ passive: true }});
            carouselContainer.addEventListener('touchmove', handleTouchMove, {{ passive: true }});
            carouselContainer.addEventListener('touchend', handleTouchEnd, {{ passive: true }});
            
            // Mouse events for desktop dragging
            carouselContainer.addEventListener('mousedown', handleMouseDown);
            carouselContainer.addEventListener('mousemove', handleMouseMove);
            carouselContainer.addEventListener('mouseup', handleMouseUp);
            carouselContainer.addEventListener('mouseleave', handleMouseUp);
            
            // Keyboard navigation
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'ArrowLeft') changeSlide(-1);
                if (e.key === 'ArrowRight') changeSlide(1);
            }});
            
            function handleTouchStart(e) {{
                startX = e.touches[0].clientX;
                isDragging = true;
            }}
            
            function handleTouchMove(e) {{
                if (!isDragging) return;
                endX = e.touches[0].clientX;
            }}
            
            function handleTouchEnd(e) {{
                if (!isDragging) return;
                isDragging = false;
                handleSwipe();
            }}
            
            function handleMouseDown(e) {{
                startX = e.clientX;
                isDragging = true;
                carouselContainer.style.cursor = 'grabbing';
            }}
            
            function handleMouseMove(e) {{
                if (!isDragging) return;
                endX = e.clientX;
            }}
            
            function handleMouseUp(e) {{
                if (!isDragging) return;
                isDragging = false;
                carouselContainer.style.cursor = 'grab';
                handleSwipe();
            }}
            
            function handleSwipe() {{
                const threshold = 50; // Minimum swipe distance
                const deltaX = startX - endX;
                
                if (Math.abs(deltaX) > threshold) {{
                    if (deltaX > 0) {{
                        changeSlide(1); // Swipe left, go to next
                    }} else {{
                        changeSlide(-1); // Swipe right, go to previous
                    }}
                }}
            }}
            
            function changeSlide(direction) {{
                const slides = document.querySelectorAll('.carousel-slide');
                const dots = document.querySelectorAll('.dot');
                
                // Hide current slide
                slides[currentSlideIndex].style.display = 'none';
                dots[currentSlideIndex].classList.remove('active');
                
                // Calculate new index
                currentSlideIndex += direction;
                
                // Handle wrap around
                if (currentSlideIndex >= totalSlides) {{
                    currentSlideIndex = 0;
                }} else if (currentSlideIndex < 0) {{
                    currentSlideIndex = totalSlides - 1;
                }}
                
                // Show new slide
                slides[currentSlideIndex].style.display = 'block';
                dots[currentSlideIndex].classList.add('active');
                
                // Update counter
                document.getElementById('current-image').textContent = currentSlideIndex + 1;
            }}
            
            function currentSlide(index) {{
                const slides = document.querySelectorAll('.carousel-slide');
                const dots = document.querySelectorAll('.dot');
                
                // Hide all slides
                slides.forEach(slide => slide.style.display = 'none');
                dots.forEach(dot => dot.classList.remove('active'));
                
                // Show selected slide
                currentSlideIndex = index - 1;
                slides[currentSlideIndex].style.display = 'block';
                dots[currentSlideIndex].classList.add('active');
                
                // Update counter
                document.getElementById('current-image').textContent = currentSlideIndex + 1;
            }}
            
            // Auto-resize carousel container
            window.addEventListener('resize', function() {{
                // Handle any resize logic if needed
            }});
        </script> 
        ```
        """
