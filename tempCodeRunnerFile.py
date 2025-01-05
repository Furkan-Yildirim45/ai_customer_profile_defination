    merged_data["order_hour_of_day"] = purchase_summary.set_index('customer_id').loc[merged_data['customer_id'], 'last_order_date'].dt.hour.values
