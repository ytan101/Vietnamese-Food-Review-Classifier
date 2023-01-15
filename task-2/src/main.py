import os
import argparse

from data_cleaning import DataCleaning
from data_processing import DataProcessing
from model import SentimentClassifier
from rich import print

# import argparse if need to choose between train, test, eval (and also for own input)

if __name__ == "__main__":
    # If clean_data doesn't exist, run data_cleaning
    if not os.path.exists("../data/cleaned/cleaned_data.csv"):
        data_cleaning = DataCleaning()
        data_cleaning.clean_data()

    parser = argparse.ArgumentParser(
        description="Get the sentiment of Vietnamese food reviews"
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="train: enables training loop, infer: predicts your input sentiment",
        default="train",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="model path, models located in ./data/raw",
        default="../data/models/model_2023-01-15_01-15-00.pt",
    )
    args = parser.parse_args()

    # Process the data
    data_processing = DataProcessing()
    reviews_dataset = data_processing.process_data()

    # Modelling
    model = SentimentClassifier(
        dataset=reviews_dataset,
        mode=args.mode,
        batch_size=8,
        truncation_length=512,
        num_epochs=2,
    )

    if args.mode == "train":
        model.training_loop()
        model.testing_loop()

    if args.mode == "infer":
        text_input = """
                        Khu ẩm thực với đa dạng đồ lại còn bày trí đẹp nên là rất hấp dẫn đối với
                        những đứa cuồng ăn uống như mình Đồ ăn giá cả phải không quá Tuy nhiên về 
                        chất lượng thì chưa thật sự ấn tượng Mình thấy ổn thôi chứ chưa thực sự có 
                        vài món để ngoài lâu nên bị nguội ăn không ngon Kem trà xanh dù nhưng mình 
                        ăn thấy vị cũng bình bánh ngọt chỉ có nhân viên thu không có nhân viên giám 
                        tự mình lấy bánh xong rồi mang qua thanh Đúng kiểu đề cao sự tự giác và trung 
                        thực của người Nhật thích điểm
                     """
        processed_text_input = data_processing.process_single(text_input)
        text_input_dict = {"Review": [processed_text_input]}
        model.inference_loop(args.model_path, text_input_dict)
