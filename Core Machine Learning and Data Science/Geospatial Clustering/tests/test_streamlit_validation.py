"""Tests for Streamlit upload validation helpers."""

from __future__ import annotations

import pandas as pd

from src.streamlit_validation import validate_upload_dataframe


def test_upload_validation_flags_missing_columns() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    errors = validate_upload_dataframe(df)
    assert errors
    assert "Missing required columns" in errors[0]


def test_upload_validation_flags_invalid_coordinates() -> None:
    df = pd.DataFrame(
        {
            "ID": ["x"],
            "Delivery_person_ID": ["D1"],
            "Delivery_person_Age": [26],
            "Delivery_person_Ratings": [4.7],
            "Restaurant_latitude": [120.0],
            "Restaurant_longitude": [77.0],
            "Delivery_location_latitude": [13.0],
            "Delivery_location_longitude": [200.0],
            "Order_Date": ["10-01-2022"],
            "Time_Orderd": ["10:00:00"],
            "Time_Order_picked": ["10:15:00"],
            "Weatherconditions": ["Sunny"],
            "Road_traffic_density": ["Low"],
            "Vehicle_condition": [2],
            "Type_of_order": ["Snack"],
            "Type_of_vehicle": ["bike"],
            "multiple_deliveries": [0],
            "Festival": ["No"],
            "City": ["Urban"],
            "Time_taken(min)": [25],
        }
    )

    errors = validate_upload_dataframe(df)
    assert any("outside global lat/lon bounds" in message for message in errors)
