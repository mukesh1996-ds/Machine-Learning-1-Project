## Custom Exception 

import os,sys


'''
try-->
exception-->

'''

class CustomException(Exception):
    def __init__(self, error_message:Exception,error_details:sys):
        self.error_message=CustomException.get_detail_error_message(error_message=error_message,
                                                                    error_details=error_details)
        
    @staticmethod
    def get_detail_error_message(error_message:Exception,error_details:sys)->str:
        _,_,exce_tb=error_details.exc_info() ## looking for third variable
        exception_block_line_number=exce_tb.tb_frame.f_lineno
        try_block_line_number=exce_tb.tb_lineno
        file_name=exce_tb.tb_frame.f_code.co_filename
        error_message=f"""
        Error Occoured In Execution of:
        [{file_name}] at
        Try Block Line Number : [{try_block_line_number}]
        and Exception Block Line Number: [{exception_block_line_number}]
        Error Message: [{error_message}]
        """
        return error_message
    
    def __str__(self):
        return self.error_message
    
    def __repr__(self):
        return CustomException.__name__.str()
