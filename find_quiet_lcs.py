import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import glob



def main():
    plt.style.use('seaborn-v0_8-colorblind')
    goodfiles = []
    maybefiles = []
    fnames = glob.glob('/Users/veraberger/.lightkurve/cache/mastDownload/TESS/*/*')
    # fnames = open('maybe_quietlcs.txt', 'r')
    # fnames = ['/Users/veraberger/.lightkurve/cache/mastDownload/TESS/tess2020266004630-s0030-0000000278055081-0195-a_fast/tess2020266004630-s0030-0000000278055081-0195-a_fast-lc.fits']
    i =0
    for file in fnames[140:]: 
        # file =file.strip()
        i +=1
        # if i % 100 == 0:
        #     print('done with ', i)
        tlc = lk.read(file)
        fname = tlc.meta['OBJECT']
        print(fname)
        """
        clippedlc = lk.LightCurve.remove_outliers(tlc, sigma_upper = 3, sigma_lower = float('inf'))
        frac = len(clippedlc)/len(tlc)
        print(frac)
        plt.plot(tlc.time.btjd, tlc.sap_flux.value, color='tab:brown')
        plt.xlabel('Time [days]')
        plt.ylabel('SAP flux [electron / s]')
        plt.title(fname)
        plt.show()
        """
        clippedlc = lk.LightCurve.remove_outliers(tlc, sigma_upper = 3, sigma_lower = float('inf'))
        frac = len(clippedlc)/len(tlc)
        if (frac > 0.8) & (np.nanmedian(clippedlc.sap_flux.value) > 5m00):
        #     goodfiles.append(file)


            fig, ax = plt.subplots(2, figsize=(15,8), sharey=True, sharex=True)
            ax[1].plot(tlc.time.btjd, tlc.sap_flux.value, color='tab:brown') #, marker='o', markersize=0.3, ls='none')
            

            
            ax[0].plot(clippedlc.time.btjd, clippedlc.sap_flux.value, color='tab:blue') #,  marker='o', markersize=0.3, ls='none')
            ax[1].set_xlabel('BTJD [days]')
            ax[1].set_ylabel('SAP flux [electron / s]')
            
            plt.title(fname+'. frac remaining after upper $\\sigma$-clip: '+str(round(frac, 2)))
        #     # print(len(tlc))
        #     # print(len(clippedlc))
        #     # print(len(clippedlc)/len(tlc))
        #     plt.xlabel('Time [days]')
        #     plt.ylabel('Flux [electron / s]')

        #     plt.savefig('/Users/veraberger/nasa/figures/quietlcs/'+fname[:-5], dpi=200)
            plt.show()
        
            judgement = input("Does this look like a good lc? ")
            if judgement == 'y':
                goodfiles.append(file)
            elif judgement == 'm': 
                maybefiles.append(file)
            elif judgement == 'break':
                with open("quietlcs1.txt", "w") as outfile:
                    outfile.write("\n".join(goodfiles))
                with open("maybe_quietlcs2.txt", "w") as outfile:
                    outfile.write("\n".join(maybefiles))
                print(i)
                return None
    #     plt.clf()
    print(goodfiles, maybefiles)
    return None
        



def remove_upper_outliers(self, s=3.0): 
        return lk.LightCurve.remove_outliers(self, sigma_upper = s, sigma_lower = float('inf'))


if __name__ == '__main__':
    main()
