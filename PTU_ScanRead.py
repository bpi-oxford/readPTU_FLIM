# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:46:33 2024

@author: narai
"""
import struct
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from tqdm import tqdm
import time

class PTUreader():
    
    """
    PTUreader() provides the capability to retrieve raw_data or image_data from a PTU file acquired using available PQ TCSPC module in the year 2019
    
    Input arguements:
    
    filename= path + filename
    print_header  = True or False
    
    Output
    
    ptu_read_raw_data     = This function reads single-photon data from the file 'name'
                            The output variables contain the followig data:
                            sync       : number of the sync events that preceeded this detection event
                            tcspc      : number of the tcspc-bin of the event
                            channel    : number of the input channel of the event (detector-number)
                            special    : marker event-type (0: photon; else : virtual photon/line_Startmarker/line_Stopmarker/framer_marker)
                            
    For example: Please see Jupyter notebook for additional info on how to get raw TTTR data
                    
    get_flim_data_stack    = This function builds a FLIM image from raw tttr_data
                             Outputs: flim_data_stack  = [numPixX numPixY numDetectors numTCSPCbins]
    
    """
    
    # Global constants
    # Define different tag types in header
    
    tag_type = dict(
    tyEmpty8      = 0xFFFF0008,
    tyBool8       = 0x00000008,
    tyInt8        = 0x10000008,
    tyBitSet64    = 0x11000008,
    tyColor8      = 0x12000008,
    tyFloat8      = 0x20000008,
    tyTDateTime   = 0x21000008,
    tyFloat8Array = 0x2001FFFF,
    tyAnsiString  = 0x4001FFFF,
    tyWideString  = 0x4002FFFF,
    tyBinaryBlob  = 0xFFFFFFFF,
    )
    
    # Dictionary with Record Types format for different TCSPC devices and corresponding T2 or T3 TTTR mode
    rec_type = dict(
        rtPicoHarpT3     = 0x00010303,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $03 (PicoHarp)
        rtPicoHarpT2     = 0x00010203,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $03 (PicoHarp)
        rtHydraHarpT3    = 0x00010304,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $04 (HydraHarp)
        rtHydraHarpT2    = 0x00010204,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $04 (HydraHarp)
        rtHydraHarp2T3   = 0x01010304,  # (SubID = $01 ,RecFmt: $01) (V2), T-Mode: $03 (T3), HW: $04 (HydraHarp)
        rtHydraHarp2T2   = 0x01010204,  # (SubID = $01 ,RecFmt: $01) (V2), T-Mode: $02 (T2), HW: $04 (HydraHarp)
        rtTimeHarp260NT3 = 0x00010305,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $05 (TimeHarp260N)
        rtTimeHarp260NT2 = 0x00010205,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $05 (TimeHarp260N)
        rtTimeHarp260PT3 = 0x00010306,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $06 (TimeHarp260P)
        rtTimeHarp260PT2 = 0x00010206,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $06 (TimeHarp260P)
        rtMultiHarpNT3   = 0x00010307,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $07 (MultiHarp150N)
        rtMultiHarpNT2   = 0x00010207,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $07 (MultiHarp150N)
    )

    def __init__(self, filename, print_header_data = False):
        
        # raw_tttr_data = False, get_image_data = True
        # if get_image_data = True then get_raw_data = False
        # else get_raw_data = True and get_image_data = False
        # Usually a user will only demand for raw or image data
                
        #Reverse mappins of tag-type and record-type dictionary
        self.tag_type_r = {j: k for k, j in self.tag_type.items()}
        self.rec_type_r = {j: k for k, j in self.rec_type.items()}
        
        self.ptu_name        = filename
        self.print_header    = print_header_data
        
        f = open(self.ptu_name, 'rb')
        self.ptu_data_string = f.read() # ptu_data_string is a string of bytes and reads all file in memory
        f.close()

        
        #Check if the input file is a valid input file
        # Read magic and version of the PTU File
        self.magic = self.ptu_data_string[:8].rstrip(b'\0')
        self.version = self.ptu_data_string[8:16].rstrip(b'\0')
        if self.magic != b'PQTTTR':
            raise IOError("This file is not a valid PTU file. ")
            exit(0)
                

        
        self.head        = {}
        
        # Read  and print header if set True
        self._ptu_read_head(self.ptu_data_string)
        
        # # Read and return Raw TTTR Data
        # self._ptu_read_raw_data()
        self.sync = None
        self.tcspc = None
        self.channel = None
        self.special = None
        self.num     = None
        self.loc     = None
        self.num_records = self.head['TTResult_NumberOfRecords']
        
        if self.print_header == True:
            return self._print_ptu_head()
        
        return None
    
    def _ptu_TDateTime_to_time(self, TDateTime):

        EpochDiff = 25569  # days between 30/12/1899 and 01/01/1970
        SecsInDay = 86400  # number of seconds in a day

        return (TDateTime - EpochDiff) * SecsInDay

    
    def _ptu_read_tags(self, ptu_data_string, offset):

        # Get the header struct as a tuple
        # Struct fields: 32-char string, int32, uint32, int64

        tag_struct = struct.unpack('32s i I q', ptu_data_string[offset:offset+48])
        offset += 48

        # Get the tag name (first element of the tag_struct)
        tagName = tag_struct[0].rstrip(b'\0').decode()

        keys = ('idx', 'type', 'value')
        tag = {k: v for k, v in zip(keys, tag_struct[1:])}

        # Recover the name of the type from tag_dictionary
        tag['type'] = self.tag_type_r[tag['type']]
        tagStringR='NA'

        # Some tag types need conversion to appropriate data format
        if tag['type'] == 'tyFloat8':
            tag['value'] = np.int64(tag['value']).view('float64')
        elif tag['type'] == 'tyBool8':
            tag['value'] = bool(tag['value'])
        elif tag['type'] == 'tyTDateTime':
            TDateTime = np.uint64(tag['value']).view('float64')
            t = time.gmtime(self._ptu_TDateTime_to_time(TDateTime))
            tag['value'] = time.strftime("%Y-%m-%d %H:%M:%S", t)

        # Some tag types have additional data
        if tag['type'] == 'tyAnsiString':
            try: tag['data'] = ptu_data_string[offset: offset + tag['value']].rstrip(b'\0').decode()
            except: tag['data'] = ptu_data_string[offset: offset + tag['value']].rstrip(b'\0').decode(encoding  = 'utf-8', errors = 'ignore')
            tagStringR=tag['data']
            offset += tag['value']
        elif tag['type'] == 'tyFloat8Array':
            tag['data'] = np.frombuffer(ptu_data_string, dtype='float', count=tag['value']/8)
            offset += tag['value']
        elif tag['type'] == 'tyWideString':
            # WideString default encoding is UTF-16.
            tag['data'] = ptu_data_string[offset: offset + tag['value']*2].decode('utf16')
            tagStringR=tag['data']
            offset += tag['value']
        elif tag['type'] == 'tyBinaryBlob':
            tag['data'] = ptu_data_string[offset: offset + tag['value']]
            offset += tag['value']

        tagValue  = tag['value']

        return tagName, tagValue, offset, tagStringR
    
    
    def _ptu_read_head(self, ptu_data_string):
    
        offset         = 16
        FileTagEnd     = 'Header_End' 
        tag_end_offset = ptu_data_string.find(FileTagEnd.encode())

        tagName, tagValue, offset, tagString  = self._ptu_read_tags(ptu_data_string, offset)
        self.head[tagName] = tagValue

        #while offset < tag_end_offset:
        while tagName != FileTagEnd:
                tagName, tagValue, offset, tagString = self._ptu_read_tags(ptu_data_string, offset)
                if tagString=='NA': self.head[tagName] = tagValue
                else: self.head[tagName] = tagString
#                 print(tagName, tagValue)

        # End of Header file and beginning of TTTR data
        self.head[FileTagEnd] = offset


    def _print_ptu_head(self): 
        #print "head" dictionary     
        print("{:<30} {:8}".format('Head ID','Value'))

        for keys in self.head:
            val = self.head[keys] 
            print("{:<30} {:<8}".format(keys, val))     
    
    def _ptu_read_raw_data(self, cnts = None):
    
        '''
        This function reads single-photon data from the file 's'

        Returns:
        sync    : number of the sync events that preceeded this detection event
        tcspc   : number of the tcspc-bin of the event
        chan    : number of the input channel of the event (detector-number)
        special : indicator of the event-type (0: photon; else : virtual photon)
        num     : counter of the records that were actually read


        '''

        if cnts is None or len(cnts) == 0:
            cnts = [0, 0]
        elif len(cnts) < 2:
            cnts = [0, cnts[0]]

        if cnts[1] > 0:
            with open(self.ptu_name, 'rb') as fid:
                fid.seek(self.head['Header_End'], 0)

                if cnts[0] > 1:
                    fid.seek(4 * (cnts[0] - 1), 1)
                
                t3records = np.fromfile(fid, dtype='uint32', count=cnts[1])
                num = t3records.size
        else:
            t3records = np.frombuffer(self.ptu_data_string, dtype='uint32', offset=self.head['Header_End'])
            num = t3records.size
            
        record_type = self.rec_type_r[self.head['TTResultFormat_TTTRRecType']]
        # num_T3records = self.head['TTResult_NumberOfRecords']
        
     
        #Next is to do T3Records formatting according to Record_type

        if record_type == 'rtPicoHarpT3':

            print('TCSPC Hardware: {}'.format(record_type[2:]))
            #   +----------------------+ T3 32 bit record  +---------------------+
            #   |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x| --> 32 bit record
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | | | | | | | | | | | | | | | | |  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  --> Sync
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | | | | |x|x|x|x|x|x|x|x|x|x|x|x|  | | | | | | | | | | | | | | | | |  --> TCSPC bin
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   |x|x|x|x| | | | | | | | | | | | |  | | | | | | | | | | | | | | | | |  --> Spectral/TCSPC input Channel
            #   +-------------------------------+  +-------------------------------+

            WRAPAROUND = 65536                                                   # After this sync counter will overflow
            sync       = np.bitwise_and(t3records, 65535)                        # Lowest 16 bits
            tcspc      = np.bitwise_and(np.right_shift(t3records, 16), 4095)     # Next 12 bits, dtime can be obtained from header
            chan       = np.bitwise_and(np.right_shift(t3records, 28),15)        # Next 4 bits 
            special    = ((chan==15)*1)*(np.bitwise_and(tcspc,15)*1)               # Last bit for special markers
            
            index      = ((chan==15)*1)*((np.bitwise_and(tcspc,15)==0)*1)        # Find overflow locations
            chan[chan==15]  = special[chan==15]
            chan[chan==1] = 0
            chan[chan==2] = 1
            chan[chan==4] = 0
            
        elif record_type == 'rtPicoHarpT2':

            print('TCSPC Hardware: {}'.format(record_type[2:]))

            #   +----------------------+ T2 32 bit record  +---------------------+
            #   |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x| --> 32 bit record
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | | | | |x|x|x|x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  --> Sync
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   |x|x|x|x| | | | | | | | | | | | |  | | | | | | | | | | | | | | | | |  --> Spectral/TCSPC input Channel
            #   +-------------------------------+  +-------------------------------+

            WRAPAROUND = 210698240                                               # After this sync counter will overflow
            sync       = np.bitwise_and(t3records, 268435455)                    # Lowest 28 bits
            tcspc      = np.bitwise_and(t3records, 15)                           # Next 4 bits, dtime can be obtained from header
            chan       = np.bitwise_and(np.right_shift(t3records, 28),15)        # Next 4 bits 
            special    = ((chan==15)*1)*np.bitwise_and(tcspc,15)                 # Last bit for special markers
            index      = ((chan==15)*1)*((np.bitwise_and(tcspc,15)==0)*1)        # Find overflow locations

        elif record_type in ['rtHydraHarpT3', 'rtHydraHarp2T3', 'rtTimeHarp260NT3', 'rtTimeHarp260PT3','rtMultiHarpNT3']:

            print('TCSPC Hardware: {}'.format(record_type[2:]))

            #   +----------------------+ T3 32 bit record  +---------------------+
            #   |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x| --> 32 bit record
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | | | | | | | | | | | | | | | | |  | | | | | | |x|x|x|x|x|x|x|x|x|x|  --> Sync
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | | | | | | | |x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x| | | | | | | | | | |  --> TCSPC bin
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | |x|x|x|x|x|x| | | | | | | | | |  | | | | | | | | | | | | | | | | |  --> Spectral/TCSPC input Channel
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   |x| | | | | | | | | | | | | | | |  | | | | | | | | | | | | | | | | |  --> Special markers
            #   +-------------------------------+  +-------------------------------+
            WRAPAROUND = 1024                                                   # After this sync counter will overflow
            sync       = np.bitwise_and(t3records, 1023)                        # Lowest 10 bits
            tcspc      = np.bitwise_and(np.right_shift(t3records, 10), 32767)   # Next 15 bits, dtime can be obtained from header
            chan       = np.bitwise_and(np.right_shift(t3records, 25),63)       # Next 8 bits 
            special    = np.bitwise_and(t3records,2147483648)>0                 # Last bit for special markers
            index      = (special*1)*((chan==63)*1)                           # Find overflow locations
            special    = (special*1)*chan

        elif record_type == 'rtHydraHarpT2':

            print('TCSPC Hardware: {}'.format(record_type[2:]))

            #   +----------------------+ T3 32 bit record  +---------------------+
            #   |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x| --> 32 bit record
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | | | | | | | |x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  --> Sync
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | |x|x|x|x|x|x| | | | | | | | | |  | | | | | | | | | | | | | | | | |  --> Spectral/TCSPC input Channel
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   |x| | | | | | | | | | | | | | | |  | | | | | | | | | | | | | | | | |  --> Special markers
            #   +-------------------------------+  +-------------------------------+
            WRAPAROUND = 33552000                                               # After this sync counter will overflow
            sync       = np.bitwise_and(t3records, 33554431)                    # Lowest 25 bits
            chan       = np.bitwise_and(np.right_shift(t3records, 25),63)       # Next 6 bits 
            tcspc      = np.bitwise_and(chan, 15)                               
            special    = np.bitwise_and(np.right_shift(t3records, 31),1)        # Last bit for special markers
            index      = (special*1) * ((chan==63)*1)                             # Find overflow locations
            special    = (special*1)*chan

        elif record_type in ['rtHydraHarp2T2', 'rtTimeHarp260NT2', 'rtTimeHarp260PT2','rtMultiHarpNT2']:

            print('TCSPC Hardware: {}'.format(record_type[2:]))

            #   +----------------------+ T3 32 bit record  +---------------------+
            #   |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x| --> 32 bit record
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | | | | | | | |x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  --> Sync
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | |x|x|x|x|x|x| | | | | | | | | |  | | | | | | | | | | | | | | | | |  --> Spectral/TCSPC input Channel
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   |x| | | | | | | | | | | | | | | |  | | | | | | | | | | | | | | | | |  --> Special markers
            #   +-------------------------------+  +-------------------------------+

            WRAPAROUND = 33554432                                               # After this sync counter will overflow
            sync       = np.bitwise_and(t3records, 33554431)                    # Lowest 25 bits
            chan       = np.bitwise_and(np.right_shift(t3records, 25),63)       # Next 6 bits 
            tcspc      = np.bitwise_and(chan, 15)                               
            special    = np.bitwise_and(np.right_shift(t3records, 31),1)        # Last bit for special markers
            index      = (special*1) * ((chan==63)*1)                            # Find overflow locations
            special    = (special*1)*chan

        else:
            print('Illegal RecordType!')
            exit(0)



        # Fill in the correct sync values for overflow location    
        #sync[np.where(index == 1)] = 1 # assert values of sync = 1 just right after overflow to avoid any overflow-correction instability in next step
        if record_type in ['rtHydraHarp2T3','rtTimeHarp260PT3','rtMultiHarpNT3']:
            sync    = sync + (WRAPAROUND*np.cumsum(index*sync)) # For HydraHarp V1 and TimeHarp260 V1 overflow corrections 
        else:
            sync    = sync + (WRAPAROUND*np.cumsum(index)) # correction for overflow to sync varibale

        sync     = np.delete(sync, np.where(index == 1), axis = 0)
        tcspc    = np.delete(tcspc, np.where(index == 1), axis = 0)
        chan     = np.delete(chan, np.where(index == 1), axis = 0)
        special  = np.delete(special, np.where(index == 1), axis = 0)
             
        # del index

        # Convert to appropriate data type to save memory

        self.sync    = sync.astype(np.uint64, copy=False)
        self.tcspc   = tcspc.astype(np.uint16, copy=False)
        self.channel = chan.astype(np.uint8,  copy=False)
        self.special = special.astype(np.uint8, copy=False)
        self.num     = num.astype(np.uint16, copy=False)   
        last_zero_index = np.where(index == 0)[0][-1] if np.any(index == 0) else -1
        self.loc = num - (last_zero_index + 1) if last_zero_index != -1 else num
        

        print("Raw Data has been Read!\n")

        return self.sync, self.tcspc, self.channel, self.special, self.num, self.loc
    
    
    
    
    def get_photon_chunk(self, start_idx, end_idx):
       """Get a chunk of photon data from start_idx to end_idx."""
       return self._ptu_read_raw_data(start_idx, end_idx)
    
    
def PTU_ScanRead(filename, plt_flag=False):
    photons = int(1e7) # number of photons to read at a time. Can be adjusted based on the system memory
    
    ptu_reader = PTUreader(filename)
    head = ptu_reader.head
    
    if not head:
        print("Header data could not be read. Aborting...")
        return None, None, None, None, None, None, None

    nx = head['ImgHdr_PixX']
    ny = head['ImgHdr_PixY']
    
  
   

    num_records = ptu_reader.num_records
    if head['ImgHdr_Ident'] in [1, 6]:  # Scan Cases 1 and 6
        anzch = 32  # max number of channels (can be 64 now for the new MultiHarp)
        
        # Common settings
        Resolution = max([1e9 * head['MeasDesc_Resolution'], 0.064])
        chDiv = 1e-9 * Resolution / head['MeasDesc_Resolution']
        Ngate = int(np.ceil(1e9 * head['MeasDesc_GlobalResolution'] / Resolution)) + 1
        head['MeasDesc_Resolution'] = Resolution * 1e-9

        # defaults
        LineStart = 4
        LineStop = 2
        
        if 'ImgHdr_LineStart' in head:
            LineStart = 2 ** (head['ImgHdr_LineStart'] - 1)
        if 'ImgHdr_LineStop' in head:
            LineStop = 2 ** (head['ImgHdr_LineStop'] - 1)
        
        y = []
        tmpx = []
        chan = []
        markers = []
        dt = np.zeros((ny,1))
        
        im_sync = []
        im_tcspc = []
        im_chan = []
        im_line = []
        im_col = []
        im_frame = []
        Turns1 = []
        Turns2 = []

        cnt = 0
        tend = 0
        line = 0 # python compatible
      
        if head['ImgHdr_BiDirect'] == 0:  # Unidirectional scan
            tmp_sync, tmp_tcspc, tmp_chan, tmp_special, num, loc = ptu_reader.get_photon_chunk([cnt+1, photons])
            while num>0:
              
                cnt += num
                if len(y)>0:
                    tmp_sync = tmp_sync + tend
                
                ind = (tmp_special>0) or ((tmp_chan<anzch) and(tmp_tcspc<Ngate*chDiv));
                
                y = np.concatenate((y, tmp_sync[ind]))  # Appending selected elements to y
                tmpx = np.concatenate((tmpx, np.floor(tmp_tcspc[ind] / chDiv) + 1))  # Appending selected elements to tmpx
                chan = np.concatenate((chan, tmp_chan[ind] + 1))  # Appending selected elements to chan
                markers = np.concatenate((markers, tmp_special[ind]))  # Appending selected elements to markers

                if LineStart == LineStop:
                    tmpturns = y[markers == LineStart]
                    if len(Turns1) > len(Turns2):
                        Turns1.extend(tmpturns[1::2])
                        Turns2.extend(tmpturns[::2])
                    else:
                        Turns1.extend(tmpturns[::2])
                        Turns2.extend(tmpturns[1::2])
                else:
                    Turns1.extend(y[markers == LineStart])
                    Turns2.extend(y[markers == LineStop])

                ind = (markers != 0)
                y = np.delete(y, ind)
                tmpx = np.delete(tmpx, ind)
                tmp_chan = np.delete(tmp_chan, ind)
                markers = np.delete(markers, ind)

                tend = y[-1] + loc
# TODO: Modify beyond this line
                if len(Turns2) > 1:
                    for j in range(len(Turns2) - 1):
                        t1 = Turns1[0]
                        t2 = Turns2[0]

                        ind = (y < t1)
                        y = np.delete(y, ind)
                        tmpx = np.delete(tmpx, ind)
                        tmp_chan = np.delete(tmp_chan, ind)

                        ind = (y >= t1) and (y <= t2)

                        im_sync.extend(y[ind])
                        im_tcspc.extend(tmpx[ind].astype(np.uint16))
                        im_chan.extend(tmp_chan[ind].astype(np.uint8))
                        im_line.extend([line].astype(np.uint16) * np.sum(ind))
                        im_col.extend(np.floor(nx * (y[ind] - t1) / (t2 - t1))) # Python compatible, pixel starts from zero

                        dt = t2 - t1
                        line += 1

                        Turns1 = Turns1[1:]
                        Turns2 = Turns2[1:]
                tmp_sync, tmp_tcspc, tmp_chan, tmp_special, num, loc = ptu_reader.get_photon_chunk([cnt+1, photons])
            
            
            t1 = Turns1[-1]
            t2 = Turns2[-1]

            ind          = (y<t1);
            y = np.delete(y, ind)
            tmpx = np.delete(tmpx, ind)
            tmp_chan = np.delete(tmp_chan, ind)

            ind = (y>=t1) and (y<=t2);

            im_sync   = [im_sync; y(ind)];
            im_tcspc  = [im_tcspc; uint16(tmpx(ind))];
            im_chan   = [im_chan; uint8(chan(ind))];
            im_line   = [im_line; uint16(line.*ones(sum(ind),1))];
            im_col    = [im_col;  uint16(1 + floor(nx.*(y(ind)-t1)./(t2-t1)))];
            dt(line)  = t2-t1;

                line = line +1;    
            head['ImgHdr_PixelTime'] = 1e9 * np.mean(dt) / nx / head['TTResult_SyncRate']
            head['ImgHdr_DwellTime'] = head['ImgHdr_PixelTime']

        else:  # Bidirectional scan
            while cnt < num_records:
                end_idx = min(cnt + photons, num_records)
                tmp_sync, tmp_tcspc, tmp_chan, tmp_special = ptu_reader.get_photon_chunk(cnt, end_idx)

                cnt += len(tmp_sync)
                ind = (tmp_chan < anzch) & (tmp_tcspc <= Ngate * chDiv)

                y = tmp_sync[ind] + tend
                tmpx = np.floor(tmp_tcspc[ind] / chDiv).astype(int) + 1
                tmp_chan = tmp_chan[ind] + 1

                markers = tmp_special[ind]
                if LineStart == LineStop:
                    tmpturns = y[markers == LineStart]
                    if len(Turns1) > len(Turns2):
                        Turns1.extend(tmpturns[1::2])
                        Turns2.extend(tmpturns[::2])
                    else:
                        Turns1.extend(tmpturns[::2])
                        Turns2.extend(tmpturns[1::2])
                else:
                    Turns1.extend(y[markers == LineStart])
                    Turns2.extend(y[markers == LineStop])

                ind = (markers != 0)
                y = np.delete(y, ind)
                tmpx = np.delete(tmpx, ind)
                tmp_chan = np.delete(tmp_chan, ind)
                markers = np.delete(markers, ind)

                tend = y[-1]

                if len(Turns2) > 2:
                    for j in range(0, 2 * (len(Turns2) // 2) - 1, 2):
                        t1 = Turns1[0]
                        t2 = Turns2[0]

                        ind = (y < t1)
                        y = np.delete(y, ind)
                        tmpx = np.delete(tmpx, ind)
                        tmp_chan = np.delete(tmp_chan, ind)
                        markers = np.delete(markers, ind)

                        ind = (y >= t1) & (y <= t2)

                        im_sync.extend(y[ind])
                        im_tcspc.extend(tmpx[ind])
                        im_chan.extend(tmp_chan[ind])
                        im_line.extend([line] * np.sum(ind))
                        im_col.extend(1 + np.floor(nx * (y[ind] - t1) / (t2 - t1)))

                        dt = t2 - t1
                        line += 1

                        t1 = Turns1[1]
                        t2 = Turns2[1]

                        ind = (y < t1)
                        y = np.delete(y, ind)
                        tmpx = np.delete(tmpx, ind)
                        tmp_chan = np.delete(tmp_chan, ind)
                        markers = np.delete(markers, ind)

                        ind = (y >= t1) & (y <= t2)

                        im_sync.extend(y[ind])
                        im_tcspc.extend(tmpx[ind])
                        im_chan.extend(tmp_chan[ind])
                        im_line.extend([line] * np.sum(ind))
                        im_col.extend(nx - np.floor(nx * (y[ind] - t1) / (t2 - t1)))

                        dt = t2 - t1
                        line += 1

                        Turns1 = Turns1[2:]
                        Turns2 = Turns2[2:]

            head['ImgHdr_PixelTime'] = 1e9 * np.mean(dt) / nx / head['TTResult_SyncRate']
            head['ImgHdr_DwellTime'] = head['ImgHdr_PixelTime']

    elif head['ImgHdr_Ident'] == 3:  # Multi-frame case (Ident = 3)
        dt = np.zeros(ny)
        f_times = []

        while cnt < num_records:
            end_idx = min(cnt + photons, num_records)
            tmp_sync, tmp_tcspc, tmp_chan, tmp_special = ptu_reader.get_photon_chunk(cnt, end_idx)

            cnt += len(tmp_sync)
            tmp_sync += tend

            y = tmp_sync
            tmpx = tmp_tcspc
            markers = tmp_special

            F = y[markers & (1 << (head['ImgHdr_Frame'] - 1)) > 0]
            while F.size > 0:
                if F.size > 0:  # Frame by Frame
                    ind = y < F[0]
                    f_y = y[ind]
                    f_x = tmpx[ind]
                    f_ch = tmp_chan[ind]
                    f_m = markers[ind]

                    y = y[~ind]
                    tmpx = tmpx[~ind]
                    tmp_chan = tmp_chan[~ind]
                    markers = markers[~ind]

                    L1 = f_y[f_m & (1 << (head['ImgHdr_LineStart'] - 1)) > 0]
                    L2 = f_y[f_m & (1 << (head['ImgHdr_LineStop'] - 1)) > 0]
                    if L1.size > 1:
                        frame += 1
                        for j in range(L2.size):
                            ind = (f_y > L1[j]) & (f_y < L2[j])

                            im_sync.extend(f_y[ind])
                            im_tcspc.extend(f_x[ind])
                            im_chan.extend(f_ch[ind])
                            im_line.extend([line] * np.sum(ind))
                            im_col.extend(1 + np.floor(nx * (f_y[ind] - L1[j]) / (L2[j] - L1[j])))
                            im_frame.extend([frame] * np.sum(ind))

                            dt[line - 1] += L2[j] - L1[j]
                            line += 1

                        f_times.append(F[0])
                        F = F[1:]

            tend = y[-1]

        head['ImgHdr_FrameTime'] = 1e9 * np.mean(np.diff(f_times)) / head['TTResult_SyncRate']
        head['ImgHdr_PixelTime'] = 1e9 * np.mean(dt) / nx / head['TTResult_SyncRate']
        head['ImgHdr_DwellTime'] = head['ImgHdr_PixelTime'] / frame

    elif head['ImgHdr_Ident'] == 9:  # Multi-frame case (Ident = 9)
        if 'ImgHdr_MaxFrames' in head:
            nz = head['ImgHdr_MaxFrames']
        else:
            time_per_frame = 1.0 / head['ImgHdr_LineFrequency'] * ny
            total_time = head['TTResult_StopAfter'] * 1e-3
            nz = int(np.ceil(total_time / time_per_frame))

        tag = np.zeros((nx, ny, len(set(ptu_reader.channel)), nz))
        tau = np.zeros_like(tag)

        while cnt < num_records:
            end_idx = min(cnt + photons, num_records)
            tmp_sync, tmp_tcspc, tmp_chan, tmp_special = ptu_reader.get_photon_chunk(cnt, end_idx)

            cnt += len(tmp_sync)
            tmp_sync += tend

            y = tmp_sync
            tmpx = tmp_tcspc
            markers = tmp_special

            Framechange = y[markers & (1 << (head['ImgHdr_Frame'] - 1)) > 0]

            if Framechange.size > 0:
                for k in range(Framechange.size):
                    line = 1
                    ind = y < Framechange[k]

                    yf = y[ind]
                    tmpxf = tmpx[ind]
                    chanf = tmp_chan[ind]

                    y = y[~ind]
                    tmpx = tmpx[~ind]
                    tmp_chan = tmp_chan[~ind]

                    Turns2f = [val for val in Turns2 if val < Framechange[k]]
                    Turns1f = [val for val in Turns1 if val < Framechange[k]]

                    Turns2 = [val for val in Turns2 if val >= Framechange[k]]
                    Turns1 = [val for val in Turns1 if val >= Framechange[k]]

                    if len(Turns2f) > 1:
                        for j in range(len(Turns2f)):
                            t1 = Turns1f[0]
                            t2 = Turns2f[0]

                            ind = (yf > t1) & (yf < t2)

                            im_frame.extend([frame] * np.sum(ind))
                            im_sync.extend(yf[ind])
                            im_tcspc.extend(tmpxf[ind])
                            im_chan.extend(chanf[ind])
                            im_line.extend([line] * np.sum(ind))
                            im_col.extend(1 + np.floor(nx * (yf[ind] - t1) / (t2 - t1)))

                            cn_phot += np.sum(ind)
                            dt[line - 1] = t2 - t1
                            line += 1

                            Turns1f = Turns1f[1:]
                            Turns2f = Turns2f[1:]

                    frame += 1

            tend = y[-1]

        head['ImgHdr_PixelTime'] = 1e9 * np.mean(dt) / nx / head['TTResult_SyncRate']
        head['ImgHdr_DwellTime'] = head['ImgHdr_PixelTime']

    else:
        print("Unsupported ImgHdr_Ident value:", head['ImgHdr_Ident'])
        return None, None, None, None, None, None, None

    # Plot if required
    if plt_flag:
        x = head['ImgHdr_X0'] + np.arange(1, nx + 1) * head['ImgHdr_PixResol']
        y = head['ImgHdr_Y0'] + np.arange(1, ny + 1) * head['ImgHdr_PixResol']

        tags = np.sum(im_sync, axis=0)
        taus = np.sum(im_tcspc, axis=0)

        plt.figure()
        plt.imshow(tags, extent=(x.min(), x.max(), y.min(), y.max()))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()
        plt.xlabel('x / µm')
        plt.ylabel('y / µm')
        plt.title('Intensity')
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.imshow(taus, extent=(x.min(), x.max(), y.min(), y.max()))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()
        plt.xlabel('x / µm')
        plt.ylabel('y / µm')
        plt.title('FLIM')
        plt.colorbar()
        plt.show()

    return head, np.array(im_sync), np.array(im_tcspc), np.array(im_chan), np.array(im_line), np.array(im_col), np.array(im_frame)
       
    