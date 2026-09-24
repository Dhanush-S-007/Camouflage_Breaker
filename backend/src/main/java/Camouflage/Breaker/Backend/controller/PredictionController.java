package Camouflage.Breaker.Backend.controller;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class PredictionController {

    private final RestTemplate restTemplate = new RestTemplate();

    @GetMapping("/health")
    public Map<String, Object> health() {
        return Map.of(
                "status", "ok",
                "service", "Camouflage Breaker Java Backend"
        );
    }

    @PostMapping(
            value = "/predict",
            consumes = MediaType.MULTIPART_FORM_DATA_VALUE,
            produces = MediaType.APPLICATION_JSON_VALUE
    )
    public ResponseEntity<?> predict(
            @RequestParam("file") MultipartFile file
    ) {

        try {

            if (file == null || file.isEmpty()) {
                return ResponseEntity.badRequest().body(
                        Map.of(
                                "success", false,
                                "error", "No file uploaded"
                        )
                );
            }

            ByteArrayResource resource =
                    new ByteArrayResource(file.getBytes()) {

                        @Override
                        public String getFilename() {
                            return file.getOriginalFilename();
                        }
                    };

            HttpHeaders fileHeaders = new HttpHeaders();

            String contentType = file.getContentType();

            if (contentType != null && !contentType.isBlank()) {
                fileHeaders.setContentType(
                        MediaType.parseMediaType(contentType)
                );
            } else {
                fileHeaders.setContentType(
                        MediaType.APPLICATION_OCTET_STREAM
                );
            }

            HttpEntity<ByteArrayResource> fileEntity =
                    new HttpEntity<>(resource, fileHeaders);

            MultiValueMap<String, Object> body =
                    new LinkedMultiValueMap<>();

            body.add("file", fileEntity);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            HttpEntity<MultiValueMap<String, Object>> request =
                    new HttpEntity<>(body, headers);

            ResponseEntity<String> response =
                    restTemplate.postForEntity(
                            "http://127.0.0.1:8000/predict",
                            request,
                            String.class
                    );

            return ResponseEntity
                    .status(response.getStatusCode())
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(response.getBody());

        } catch (Exception e) {

            String message = e.getMessage();

            if (message == null || message.isBlank()) {
                message = "Unknown error";
            }

            return ResponseEntity
                    .status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(
                            Map.of(
                                    "success", false,
                                    "error", "AI service connection failed",
                                    "message", message
                            )
                    );
        }
    }
}