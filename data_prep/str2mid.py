from mido import Message, MidiFile, MidiTrack
import sys
import getopt

def main(argv):
    # script -i <inputfile> -o <outputfile>
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('str2mid.py -i <inputfile> -o <outputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('str2mid.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    if len(outputfile) == 0:
        filename = inputfile.split('/')[-1]
        filename = filename.split('.')[0]
        outputfile = '/home/kell/PycharmProject/Pianissimo/model' + filename + '.mid'

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(Message('program_change', program=12, time=0))

    with open(inputfile, 'r') as f:

        content = f.readlines()

    content = [list(map(int, x.strip().split(','))) for x in content]
    for pack in content:
        i = 0
        for x in pack:
            track.append(Message('note_on', note=x + 60, velocity=64, time=0))
        for x in pack:
            if i == 0:
                track.append(Message('note_on', note=x + 60, velocity=0, time=480))
            track.append(Message('note_on', note=x + 60, velocity=0, time=0))
            i += 1

    mid.save(outputfile)

if __name__ == "__main__":
    main(sys.argv[1:])
