using SixLabors.ImageSharp.Processing;
using SwarmUI.Utils;
using SwarmUI.Media;
using Image = SwarmUI.Utils.Image;
using ISImage = SixLabors.ImageSharp.Image;
using ISImage32 = SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>;
using ISImageFrame32 = SixLabors.ImageSharp.ImageFrame<SixLabors.ImageSharp.PixelFormats.Rgba32>;

namespace Hartsy.Extensions.MagicPromptExtension;

public static class BackendSchema
{
    public enum MessageType
    {
        Text,
        Vision
    }

    public class MessageContent
    {
        public string Text { get; set; }
        public string Instructions { get; set; }
        public List<MediaContent> Media { get; set; }
        public int? KeepAlive { get; set; }
    }

    public class MediaContent
    {
        public string Type { get; set; }  // "base64" or "url"
        public string Data { get; set; }
        public string MediaType { get; set; }  // "image/jpeg", "image/png", etc.
    }

    /// <summary>Get the schema type for the backend.</summary>
    /// <param name="type">Backend type (ollama, openai, anthropic, etc.)</param>
    /// <param name="content">Message content including text and media</param>
    /// <param name="model">Model name to use</param>
    /// <param name="messageType">Type of message (Text or Vision)</param>
    /// <returns>Returns an object with the schema type for the backend.</returns>
    public static object GetSchemaType(string type, MessageContent content, string model, MessageType messageType = MessageType.Text, long seed = -1)
    {
        if (content == null || string.IsNullOrEmpty(model))
        {
            throw new ArgumentException("Content or model cannot be null or empty.");
        }
        type = type.ToLower();
        _ = content.KeepAlive;
        return type switch
        {
            "ollama" => OllamaRequestBody(content, model, messageType, seed),
            "grok" => OpenAICompatibleRequestBody(content, model, messageType, preferPngForBase64: true, seed),
            "openai" or "openaiapi" or "openrouter" => OpenAICompatibleRequestBody(content, model, messageType, preferPngForBase64: false, seed),
            "anthropic" => AnthropicRequestBody(content, model, messageType),
            _ => throw new ArgumentException($"Unsupported backend type: {type}")
        };
    }

    /// <summary>Compresses image data to optimize for LLM vision models</summary>
    /// <param name="media">The media content containing image data</param>
    /// <param name="targetFormat">The target format ("PNG", "JPG", or "WEBP")</param>
    /// <returns>Compressed base64 image data (without a data URL prefix) together with its resulting MIME type</returns>
    public static (string Data, string MimeType) CompressImageForVision(MediaContent media, string targetFormat = "WEBP")
    {
        if (media.Type != "base64")
        {
            return (media.Data, media.MediaType);
        }
        try
        {
            ImageFile image = ImageFile.FromDataString($"data:{media.MediaType};base64,{media.Data}");
            // Skip compression for videos etc..
            if (image.Type.MetaType != MediaMetaType.Image)
            {
                return (media.Data, media.MediaType);
            }
            ISImage img = image.ToIS;
            // Fix (Claude, 2026-07-27): 256px was too aggressive for modern higher-resolution vision
            // encoders (confirmed via direct testing: the exact same image reliably misidentified at
            // 256px/quality-40 was reliably correct at full resolution against the same model). 1024px
            // preserves much more real detail while still keeping payload size reasonable.
            int maxDimension = 1024;
            if (img.Width > maxDimension || img.Height > maxDimension)
            {
                float scaleFactor = maxDimension / (float)Math.Max(img.Width, img.Height);
                int newWidth = (int)(img.Width * scaleFactor);
                int newHeight = (int)(img.Height * scaleFactor);
                img.Mutate(i => i.Resize(newWidth, newHeight));
            }
            // Fix (Claude, 2026-07-27): quality 40/60 was heavily lossy on top of the aggressive
            // downscale above; 90 preserves detail much better at a modest size cost.
            int quality = 90;
            ImageFile tempImage = new Image(ImageFile.ISImgToPngBytes(img), image.Type);
            ImageFile compressedImage = tempImage.ConvertTo(targetFormat, quality: quality);
            // Fix (CodeRabbit review, 2026-07-27): report the actual resulting MIME type instead of
            // letting callers assume one from targetFormat - the fallback paths above return the
            // original untouched bytes on non-image media or a conversion failure, so callers need
            // to know that happened in order to label the data URL correctly.
            string resultMimeType = targetFormat switch
            {
                "PNG" => "image/png",
                "JPG" => "image/jpeg",
                "WEBP" => "image/webp",
                _ => media.MediaType
            };
            return (compressedImage.AsBase64, resultMimeType);
        }
        catch (Exception ex)
        {
            Logs.Error($"Failed to compress image: {ex.Message}");
            return (media.Data, media.MediaType);
        }
    }

    /// <summary>Generates a request body for Ollama backend.</summary>
    private static object OllamaRequestBody(MessageContent content, string model, MessageType messageType, long seed = -1)
    {
        List<object> messages = [];
        if (!string.IsNullOrEmpty(content.Instructions))
        {
            messages.Add(new { role = "system", content = content.Instructions });
        }

        object options = seed == -1
            ? new { temperature = 1.0, top_p = 0.9 }
            : new { temperature = 1.0, top_p = 0.9, seed };

        if (messageType == MessageType.Vision && content.Media?.Any() == true)
        {
            messages.Add(new
            {
                role = "user",
                content = content.Text,
                images = content.Media.Select(m => CompressImageForVision(m, "JPG").Data).ToArray()
            });

            return new
            {
                model,
                messages = messages.ToArray(),
                stream = false,
                keep_alive = content.KeepAlive,
                options
            };
        }
        messages.Add(new { role = "user", content = content.Text });
        return new
        {
            model,
            messages = messages.ToArray(),
            stream = false,
            keep_alive = content.KeepAlive,
            options
        };
    }

    /// <summary>Generates a request body for OpenAI and compatible backends.</summary>
    private static object OpenAICompatibleRequestBody(MessageContent content, string model, MessageType messageType, bool preferPngForBase64, long seed = -1)
    {
        List<object> messages = [];
        // Add system message if instructions exist
        if (!string.IsNullOrEmpty(content.Instructions))
        {
            messages.Add(new { role = "system", content = content.Instructions });
        }
        if (messageType == MessageType.Vision && content.Media?.Any() == true)
        {
            List<object> contentList = [];
            foreach (MediaContent media in content.Media)
            {
                // Fix (Claude, 2026-07-27): WEBP was confirmed (via direct A/B testing, same image,
                // format as the only variable) to be mishandled by at least one real vision model's
                // decode pipeline even at near-lossless quality - the model would confidently describe
                // unrelated content. JPEG was confirmed reliable in the same test and is what Ollama's
                // own request-building code already uses (CompressImageForVision(m, "JPG") above),
                // so this brings the OpenAI-compatible path in line with that already-proven choice.
                (string imageData, string imageMimeType) = CompressImageForVision(media, preferPngForBase64 ? "PNG" : "JPG");
                contentList.Add(new
                {
                    type = "image_url",
                    image_url = media.Type == "base64"
                        ? new { url = $"data:{imageMimeType};base64,{imageData}" }
                        : new { url = media.Data }
                });
            }
            contentList.Add(new
            {
                type = "text",
                text = content.Text
            });
            messages.Add(new
            {
                role = "user",
                content = contentList
            });

            if (seed != -1)
            {
                return new
                {
                    model,
                    messages = messages.ToArray(),
                    max_tokens = 1000,
                    // Fix (Claude, 2026-07-27): vision/captioning wants low, near-deterministic sampling,
                    // not the creative-chat default of 1.0 - confirmed via testing that high temperature
                    // turns a marginal image into confidently-wrong, differently-wrong-each-time answers.
                    temperature = 0.2,
                    stream = false,
                    seed
                };
            }

            return new
            {
                model,
                messages = messages.ToArray(),
                max_tokens = 1000,
                temperature = 0.2,
                stream = false
            };
        }
        messages.Add(new { role = "user", content = content.Text });

        if (seed != -1)
        {
            return new
            {
                model,
                messages = messages.ToArray(),
                max_tokens = 1000,
                temperature = 1.0,
                stream = false,
                seed
            };
        }

        return new
        {
            model,
            messages = messages.ToArray(),
            temperature = 1.0,
            max_tokens = 1000,
            top_p = 0.9,
            stream = false
        };
    }

    /// <summary>Generates a request body for the Anthropic (Claude) API.</summary>
    private static object AnthropicRequestBody(MessageContent content, string model, MessageType messageType)
    {
        List<object> messages = [];
        if (messageType == MessageType.Vision && content.Media?.Any() == true)
        {
            List<object> messageContent = [];
            foreach (MediaContent media in content.Media)
            {
                // Compress image and convert to PNG. Anthropic only accepts PNG.
                (string imageData, string mediaType) = CompressImageForVision(media, "PNG");
                messageContent.Add(new
                {
                    type = "image",
                    source = new
                    {
                        type = "base64",
                        media_type = mediaType,
                        data = imageData
                    }
                });
            }
            messageContent.Add(new
            {
                type = "text",
                text = content.Text
            });
            messages.Add(new
            {
                role = "user",
                content = messageContent.ToArray()
            });
            return new
            {
                model,
                messages = messages.ToArray(),
                system = content.Instructions,
                max_tokens = 1024
            };
        }
        messages.Add(new { role = "user", content = content.Text });
        return new
        {
            model,
            messages = messages.ToArray(),
            system = content.Instructions,
            max_tokens = 1024
        };
    }
}