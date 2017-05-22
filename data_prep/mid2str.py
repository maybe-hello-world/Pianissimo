#!/usr/bin/python3
from mido import MidiFile
import sys
import getopt


# Class for MIDI notes
class Note:
    def __init__(self, action, note, time):
        self.action = action
        self.note = int(note)
        self.time = int(time)

    def __repr__(self):
        return "action=" + self.action + " note=" + str(self.note) + " time=" + str(self.time)

    def action(self):
        return self.action

    def note(self):
        return self.note

    def time(self):
        return self.time


def main(argv):
    # script -i <inputfile> -o <outputfile>
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('mid2str.py -i <inputfile> -o <outputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('mid2str.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    if len(outputfile) == 0:
        filename = inputfile.split('/')[-1]
        filename = filename.split('.')[0]
        outputfile = '/home/kell/parsed_music/' + filename + '.txt'

    mid = MidiFile(inputfile)

    # Parse midi to notes
    notes = []
    acc_str = {'note_on', 'note_off'}
    for i, track in enumerate(mid.tracks):
        for msg in track:
            s = str(msg).split(' ')
            if s[0] in acc_str:
                if s[3] == 'velocity=0':
                    s[0] = 'note_off'
                notes.append(Note(s[0], s[2].split('=')[1], s[4].split('=')[1]))
                # print(s)

    note_packs = []
    pack = []

    for x in notes:
        if x.action == "note_off":
            if len(pack) > 0:
                note_packs.append(list(set(pack)))
            pack = []
        else:
            pack.append(int(x.note) % 12)

    with open(outputfile, 'w') as f:
        for x in note_packs:
            f.write(str(x)[1:-1] + "\n")

if __name__ == "__main__":
    main(sys.argv[1:])
