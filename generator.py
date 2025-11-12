import tensorflow as tf
import pathlib


class Generator:
    def __init__(self, source_dir, save_dir, flip=True, adjust_brightness=True, adjust_contrast=True, adjust_saturation=True, add_Gauss_noise=True):
        self.source_dir = pathlib.Path(source_dir)
        self.save_dir = pathlib.Path(save_dir)
        self.flip = flip
        self.adjust_brightness = adjust_brightness
        self.adjust_contrast = adjust_contrast
        self.adjust_saturation = adjust_saturation
        self.add_Gauss_noise = add_Gauss_noise
        # Tạo thư mục save_dir nếu chưa tồn tại
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def load_image(self, source_path):
        img = tf.io.read_file(str(source_path))
        decoded_img = tf.image.decode_image(img, channels = 3)
        decoded_img = tf.image.convert_image_dtype(decoded_img, tf.float32)
        return decoded_img
    
    def save_image(self, image_tensor, save_path):
        img = tf.image.convert_image_dtype(image_tensor, tf.uint8, saturate=True)
        encoded_img = tf.io.encode_jpeg(img, quality=95)
        tf.io.write_file(str(save_path), encoded_img)

    def process(self, number_of_img=10):
        img_paths = list(self.source_dir.glob("**/*.jpg"))
        if not img_paths:
            print("Không tìm thấy ảnh")
            return
        
        if number_of_img > len(img_paths):
            print("Quá nhiều ảnh")
            return
        
        total_saved = 0

        for img_path in img_paths:
            if total_saved == number_of_img: 
                break

            img = self.load_image(img_path)

            # Lật ảnh
            if self.flip: 
                img = tf.image.flip_left_right(img)
            
            # Điều chỉnh độ sáng
            if self.adjust_brightness: 
                img = tf.image.random_brightness(img, max_delta=0.5)

            # Điều chỉnh độ tương phản
            if self.adjust_contrast:
                img = tf.image.random_contrast(img, lower=0, upper=2)

            # Điều chỉnh độ bão hòa
            if self.adjust_saturation:
                img = tf.image.random_saturation(img, lower=0, upper=2)

            # Thêm nhiễu
            if self.add_Gauss_noise:
                noise = tf.random.normal(shape=tf.shape(img), stddev=0.1)
                img = tf.clip_by_value(img + noise, 0, 1)
            
            new_filename = f"new_{total_saved+1}.jpg"
            save_path = self.save_dir / new_filename
            self.save_image(img, save_path)
            total_saved += 1


if __name__ == "__main__":
    generator = Generator(source_dir="E:/Code/Project ML/Radiographs/Radiographs", save_dir="E:/Code/Project ML/test")
    generator.process()